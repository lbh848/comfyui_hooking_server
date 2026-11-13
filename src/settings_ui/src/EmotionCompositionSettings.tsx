import { useEffect, useRef, useState } from "react";
import { RotateCcw } from "lucide-react";
import { emotionIllustrationRequest, useEmotionIllustrationResource } from "./emotionIllustration";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { useWorkflowAutosave } from "./useWorkflowAutosave";
import { EmotionCompositionPreview } from "./EmotionCompositionPreview";

export type CharacterPosition = "R" | "M" | "L";
export type CompositionSettings = {
  visible_width: number;
  visible_height: number;
  visible_center_x: number | null;
  visible_center_y: number | null;
  character_height: number;
  character_align_x: number;
  character_align_y: number;
  character_offset_x: number;
  character_offset_y: number;
  character_positions: Record<CharacterPosition, number>;
};

export function EmotionCompositionSettings({ botSelected, onDirtyChange, onBusyChange }: {
  botSelected: string;
  onDirtyChange: (dirty: boolean) => void; onBusyChange: (busy: boolean) => void;
}) {
  const resource = useEmotionIllustrationResource("illustration/config");
  const [draft, setDraft] = useState<CompositionSettings | null>(null);
  const [baseline, setBaseline] = useState<CompositionSettings | null>(null);
  const form = useRef<HTMLFormElement>(null);
  useEffect(() => {
    if (typeof resource.data?.config?.composition_defaults?.character_positions?.M !== "number") return;
    setDraft(resource.data.config.composition_settings);
    setBaseline(resource.data.config.composition_settings);
  }, [resource.data]);
  const dirty = !!draft && JSON.stringify(draft) !== JSON.stringify(baseline);
  const autosave = useWorkflowAutosave({
    label: "감정 삽화 · 배경 합성 설정", value: JSON.stringify(draft), dirty, form,
    save: async () => {
      if (!draft) return;
      const saved = await emotionIllustrationRequest("illustration/config", {
        composition_settings: draft,
      });
      setBaseline(saved.config.composition_settings);
    },
  });
  useEffect(() => { onDirtyChange(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => { onBusyChange(autosave.busy); }, [autosave.busy, onBusyChange]);
  useEffect(() => () => { onDirtyChange(false); onBusyChange(false); }, [onDirtyChange, onBusyChange]);

  const unsupported = !!resource.data && typeof resource.data.config?.composition_defaults?.character_positions?.M !== "number";
  if (!draft) return <IllustrationLoading error={unsupported ? "합성 설정을 사용하려면 프로그램을 다시 시작하세요." : resource.error} retry={resource.reload} />;
  return <form ref={form} className="illust-form" onSubmit={event => { event.preventDefault(); void autosave.retry(); }}>
    <p className="field-help">배경에서 보일 영역의 크기와 R / M / L 각각의 인물 위치를 설정합니다. 포인트를 가로로 드래그해 조절하고 아래에서 랜덤 합성 결과를 확인하세요.</p>
    <EmotionCompositionPreview settings={draft} botSelected={botSelected} validation={autosave.inputError}
      onChange={patch => setDraft(current => current && ({ ...current, ...patch }))} />
    <div className="illust-actions"><button type="button" className="button" disabled={autosave.busy || !resource.data?.config?.composition_defaults}
      onClick={() => setDraft({ ...resource.data!.config.composition_defaults })}>
      <RotateCcw size={14} />기본값으로 복원
    </button><span role="status">{autosave.status}</span></div>
    <IllustrationFeedback error={autosave.error || autosave.inputError} />
    {autosave.error && <button className="button" disabled={autosave.busy}>다시 저장</button>}
  </form>;
}
