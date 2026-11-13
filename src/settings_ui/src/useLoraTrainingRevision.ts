import { useEffect, useState } from "react";
import { isProjectTraining, type LoraContext } from "./internalLora";
import { watchCompletedQueueItems } from "./workQueue";

export function useLoraTrainingRevision({ category, project }: Pick<LoraContext, "category" | "project">): number {
  const [revision, setRevision] = useState(0);
  const { name, character, bot, id } = project;
  useEffect(() => watchCompletedQueueItems(
    item => isProjectTraining(item, { category, project: { name, character, bot, id } }),
    () => setRevision(value => value + 1),
    cause => console.error("[LORA] 학습 완료 상태 조회 실패:", cause),
  ), [category, name, character, bot, id]);
  return revision;
}
