import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { useIllustrationAction, type IllustrationData } from "./illustration";
import { useIllustrationSaveReport } from "./IllustrationSaveStatus";
import { illustrationModelLabel, illustrationPresetFamily } from "./illustrationModels";
import { useModelSettingsRequest, useModelSettingsResource, useModelSettingsScope } from "./ModelSettingsScope";
import { SearchableSelect } from "./SearchableSelect";

function useModelSelectionState() {
  const scope = useModelSettingsScope();
  const request = useModelSettingsRequest();
  const resource = useModelSettingsResource("presets");
  const action = useIllustrationAction();
  const [data, setData] = useState<IllustrationData | null>(null);
  const [pending, setPending] = useState(false);
  const [generation, setGeneration] = useState(0);
  useEffect(() => {
    if (resource.data) { setData(resource.data); setGeneration(value => value + 1); }
  }, [resource.data]);
  // Reload presets only on explicit actions. A focus refresh increments generation
  // and remounts the workflow editor, losing its tabs, scroll, and unsaved input.
  useIllustrationSaveReport({ busy: action.busy, error: action.error || resource.error, savedAt: data?.saved_at });
  const disabled = pending || action.busy || !resource.data;
  async function select(targetId: string) {
    if (!data || disabled || targetId === data.active_id) return;
    await action.run(async () => {
      const result = await request(`${scope.prefix}/presets`, {
        action: "select", preset_id: data.active_id, revision: data.revision, target_id: targetId,
      });
      setData(result);
    }, "그림 모델 프리셋을 적용했습니다.");
  }
  // Sort a copy for both pickers without changing the server's preset order.
  const presets = [...(data?.presets || [])].sort((a: IllustrationData, b: IllustrationData) => a.name.localeCompare(b.name, "ko", { numeric: true }));
  return { resource, action, data, setData, presets, pending, setPending, generation, disabled, select };
}

const ModelSelectionContext = createContext<ReturnType<typeof useModelSelectionState> | null>(null);
export function IllustrationModelSelectionProvider({ children }: { children: ReactNode }) {
  const value = useModelSelectionState();
  return <ModelSelectionContext.Provider value={value}>{children}</ModelSelectionContext.Provider>;
}
export function useIllustrationModelSelection() {
  const value = useContext(ModelSelectionContext);
  if (!value) throw new Error("IllustrationModelSelectionProvider is required.");
  return value;
}

export function ActiveIllustrationModel() {
  const scope = useModelSettingsScope();
  const { data, presets, disabled, select, pending } = useIllustrationModelSelection();
  const profile = data?.config?.[scope.typeKey];
  const activePreset = (data?.presets || []).find((item: IllustrationData) => item.id === data?.active_id);
  // MOSS presets share the ANIMA workflow profile, so show the selected model
  // family instead of the profile label.
  const modelLabel = scope.familyKey && illustrationPresetFamily(activePreset) === "moss"
    ? "MOSS" : profile ? scope.labels[profile] || illustrationModelLabel(profile) : "";
  return <label className="field">현재 활성화된 그림 모델
    <SearchableSelect aria-label="현재 활성화된 그림 모델" value={data?.active_id || ""} disabled={disabled || !presets.length} onChange={event => void select(event.target.value)}>
      {!presets.length && <option value="">{data ? "그림 모델 프리셋 없음" : "그림 모델을 불러오는 중…"}</option>}
      {["builtin", "local"].map(group => presets.some((item: IllustrationData) => item.scope === group) && <optgroup key={group} label={group === "builtin" ? "배포 프리셋" : "사용자 프리셋"}>
        {presets.filter((item: IllustrationData) => item.scope === group).map((item: IllustrationData) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </optgroup>)}
    </SearchableSelect>
    <span className="field-help">{pending ? "모델 설정을 저장한 뒤 전환할 수 있습니다." : profile ? `그림 모델 · ${modelLabel}` : `다음 ${scope.title} 생성에 사용할 프리셋을 선택하세요.`}</span>
  </label>;
}
