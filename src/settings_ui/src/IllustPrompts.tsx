import { useEffect, useRef, useState } from "react";
import { Badge, Panel, SettingsTabs } from "./components";
import { illustrationRequest, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";
import { emotionIllustrationRequest, emotionScopePath } from "./emotionIllustration";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { IllustGenerationLogs, IllustHistory } from "./IllustHistory";
import { IllustPromptStructure } from "./IllustInspector";
import { IllustNodeFlow } from "./IllustNodeFlow";
import { IllustPromptEditor } from "./IllustPromptEditor";
import { EmotionPresetSettings, EMOTION_PRESET_CHAR_LIMIT } from "./EmotionPresetSettings";
import { IllustPostprocess } from "./IllustPostprocess";
import { useWorkflowAutosave } from "./useWorkflowAutosave";
import { useUnsavedChanges } from "./unsavedChanges";
import { illustrationPresetFamily } from "./illustrationModels";

const labels: Record<string, string> = {
  local_llm_load_after_generation: "생성이 끝나면, 공방-LLM 로드", image_model_load_at_detail: "DETAIL이 시작할 때 그림 모델 로드",
  illustration_model_family: "그림 모델",
  illustration_output_mode: "출력 모드", original_asset_count: "원본 에셋 출력 장수",
  moss_dialogue_repair_enabled: "말풍선 한국어 교정",
  manga_disable_bubbles: "말풍선 끄기",
  outfit_enabled: "OUTFIT 복장·헤어 추적", time_place_enabled: "TIME & PLACE 시간·장소·조명 추적", npc_generation_enabled: "미등록 인물 자동 생성", npc_generation_deferred: "NPC 생성은 나중에", npc_lifetime: "미사용 수명 (회)", call2_plan_enabled: "PLAN 시각 계획",
  call1_context_turns: "OUTFIT 참조 대화 수", call2_context_turns: "DETAIL 참조 대화 수", call3_context_turns: "POST PROCESS 참조 대화 수",
  single_char_mask_enabled: "1인도 마스크",
  single_char_mask_skip_enabled: "면적 기준으로 1인 마스크 생략", single_char_mask_skip_area_percent: "생략 기준 면적 (%)",
  speak_enabled: "대사 생성", call3_prompt_mode: "대사 출력 방식", speak_language: "대사 언어", speak_emotion_enabled: "대사 감정 지정", speak_emotions: "대사 감정",
  nsfw: "NSFW 태깅", overthinking_prevention_enabled: "과추론 방지", manga_composition_hint_enabled: "만화 구성 힌트", nsfw_asset_generation_enabled: "NSFW 에셋 생성", profile_context_translate: "프로필 영어 컨텍스트 번역 (PROFILE-CONTEXT-TRANSLATE)", supplement: "SUPPLEMENT", key_visual: "KEYVIS 대표 이미지 생성",
  character_limit: "이미지당 최대 캐릭터", scene_mode: "장면 수 결정", output_count_min: "최소 일반 장면 수", output_count_max: "최대 일반 장면 수",
  context_history: "이전 대화 참조", focus: "집중 캐릭터", direction: "우선 연출 지시", manga_output_count_min: "최소 만화 페이지 수", manga_output_count_max: "최대 만화 페이지 수",
  manga_scene_mode: "페이지 수 결정", manga_character_limit: "페이지당 최대 캐릭터", manga_focus: "만화 집중 캐릭터", manga_direction: "만화 우선 연출 지시",
  postprocess_enabled: "POST PROCESS", prompt_format: "프롬프트 입력 형식",
  positive_note: "긍정 프롬프트 추가", negative_note: "부정 프롬프트 추가", compat_comfy: "ComfyUI 호환", compat_character_prompt: "캐릭터 프롬프트 구성",
};
for (const [suffix, text] of Object.entries({ enabled: "병렬 처리", max_concurrency: "최대 동시 처리 수", slow_retry_enabled: "느린 요청 재시도", slow_retry_remaining: "재시도 잔여 작업 수", slow_retry_progress_enabled: "진행률 기준 사용", slow_retry_progress_threshold: "재시도 진행률 기준 (%)", slow_retry_tps_enabled: "생성 속도 기준 사용", slow_retry_tps_threshold: "재시도 생성 속도 기준", slow_retry_condition_operator: "재시도 조건 결합" })) {
  labels[`call2_parallel_${suffix}`] = `PLAN ${text}`;
}
const choices: Record<string, [string, string][]> = {
  illustration_output_mode: [["illustration", "일반 삽화"], ["original_asset", "원본 에셋"], ["both", "일반 삽화 + 원본 에셋"], ["manga", "만화 삽화"], ["manga_both", "만화 삽화 + 원본 에셋"]],
  scene_mode: [["manual", "지정 범위 적용"], ["auto", "범위 안에서 자동 선택"]], manga_scene_mode: [["manual", "지정 범위 적용"], ["auto", "범위 안에서 자동 선택"]],
  character_limit: [["1", "1"], ["2", "2"], ["3", "3"]], manga_character_limit: [["1", "1"], ["2", "2"], ["3", "3"]],
  call3_prompt_mode: [["speak", "대사"], ["manga", "말풍선"], ["subtitle", "자막"]],
  compat_character_prompt: [["separate", "캐릭터별 분리"], ["combined", "통합"]],
  speak_language: [["한국어", "한국어"], ["영어", "영어"], ["일어", "일어"], ["중어", "중어"]],
  prompt_format: [["v3", "V3"], ["chansub", "챈섭 · Comfy POSITIVE/NEGATIVE"]],
};
// Composite modes are configured in the emotion illustration workshop and are
// not exposed in this tab at all.
const compositeOutputModes = ["composite", "composite_both"];
// 감정 삽화 공방은 일반 감정 삽화 생성과 배경 합성 파이프라인 두 가지만 노출한다.
const emotionOutputChoices: [string, string][] = [
  ["illustration", "감정 삽화 모드"],
  ["composite", "배경 합성 모드"],
];
// 감정 삽화 공방의 에셋 개수 설정 탭은 '일반 장면' 대신 '장면'으로 표기한다.
const emotionAssetLabels: Record<string, string> = {
  output_count_min: "최소 장면 수",
  output_count_max: "최대 장면 수",
};
const emotionAssetHelp: Record<string, string> = {
  output_count_min: "생성할 감정 에셋 수의 하한입니다.",
  output_count_max: "생성할 감정 에셋 수의 상한입니다.",
};
type GenerationSection = { title: string; keys: string[]; dependent?: string[]; advanced?: string[]; advancedTitle?: string; paired?: boolean; mode?: "illustration" | "manga" };
const planParallelSections: GenerationSection[] = [{
  title: "",
  keys: ["call2_parallel_enabled", "call2_parallel_max_concurrency"],
  advancedTitle: "PLAN 느린 배치 재요청",
  advanced: ["enabled", "remaining", "progress_enabled", "progress_threshold", "tps_enabled", "tps_threshold", "condition_operator"].map(suffix => `call2_parallel_slow_retry_${suffix}`),
}];
const generationGroups: { id: string; title: string; description: string; sections: GenerationSection[] }[] = [
  {
    id: "pipeline", title: "파이프라인", description: "출력 방식과 생성 단계의 사용 여부를 설정합니다.",
    sections: [
      { title: "", keys: ["illustration_model_family", "illustration_output_mode", "moss_dialogue_repair_enabled", "nsfw", "overthinking_prevention_enabled", "manga_composition_hint_enabled", "manga_disable_bubbles", "profile_context_translate", "outfit_enabled", "time_place_enabled", "call2_plan_enabled"] },
      { title: "NPC 생성", keys: ["npc_generation_enabled"], dependent: ["npc_lifetime"] },
      { title: "", keys: ["key_visual", "postprocess_enabled"] },
    ],
  },
  {
    id: "original_asset", title: "에셋 선택", description: "원본 에셋을 포함하는 출력 모드에서 사용할 장수를 설정합니다. 선택 지침은 현재 봇의 에셋·프로필 출력 지침을 사용합니다.",
    sections: [{ title: "", keys: ["original_asset_count"] }],
  },
  {
    id: "selection", title: "삽화·만화 장면 선택", description: "출력 모드에 맞는 설정을 사용합니다. 삽화와 만화의 장면 설정은 각각 유지됩니다.",
    sections: [
      { title: "삽화 장면 선택", mode: "illustration", keys: ["output_count_min", "output_count_max", "scene_mode", "character_limit", "focus", "direction"] },
      { title: "만화 장면 선택", mode: "manga", keys: ["manga_output_count_min", "manga_output_count_max", "manga_scene_mode", "manga_character_limit", "manga_focus", "manga_direction"] },
    ],
  },
  {
    id: "plan_parallel", title: "PLAN 병렬처리", description: "PLAN의 병렬 처리와 느린 요청 재시도를 설정합니다.",
    sections: planParallelSections,
  },
  {
    id: "postprocess", title: "POST PROCESS", description: "대사의 언어·감정과 자막·텍스트 교체 합성을 설정합니다.",
    sections: [
      { title: "", keys: ["speak_language", "speak_emotion_enabled", "speak_emotions"] },
    ],
  },
  {
    id: "output", title: "최종 프롬프트", description: "이미지 생성에 전달할 프롬프트 형식과 추가 지침을 설정합니다.",
    sections: [
      { title: "", keys: ["prompt_format"] },
      { title: "프롬프트 추가 메모", keys: ["positive_note", "negative_note"], paired: true },
    ],
  },
];
const generationTabs = Object.fromEntries(generationGroups.map(group => [group.id, group.title]));
// 감정 삽화 공방은 OUTFIT·PLAN·KEYVIS·MULTI-CHAR-MASK 노드를 사용하지 않으므로
// 해당 생성 설정을 노출하지 않는다. DETAIL 설정은 그대로 유지한다.
const emotionHiddenKeys = new Set([
  "illustration_model_family",
  "moss_dialogue_repair_enabled",
  "manga_composition_hint_enabled",
  "manga_disable_bubbles",
  "outfit_enabled", "call2_plan_enabled", "key_visual", "keyvis_in_detail_enabled",
  "minimal_background_description",
  "single_char_mask_enabled", "single_char_mask_skip_enabled", "single_char_mask_skip_area_percent",
  // 감정 삽화는 인스턴스 에셋 생성이 목적이므로 대사 후처리를 사용하지 않는다.
  "postprocess_enabled",
]);
const emotionHiddenSections = new Set(["KEYVIS", "MULTI-CHAR-MASK"]);
// 장면 선택·PLAN 병렬처리·POST PROCESS·최종 프롬프트·원본 에셋은 감정 삽화
// 설정에서 사용하지 않는다. 파이프라인 설정만 남긴다.
const emotionHiddenGroups = new Set(["selection", "plan_parallel", "postprocess", "output", "original_asset"]);
const emotionGenerationGroups = [
  ...generationGroups.filter(group => !emotionHiddenGroups.has(group.id)).map(group => ({
    ...group,
    sections: group.sections
      .filter(section => !emotionHiddenSections.has(section.title))
      .map(section => {
        // Keep the emotion asset toggle after the shared NSFW and reasoning controls.
        const keys = section.keys
          .filter(key => !emotionHiddenKeys.has(key))
          .flatMap(key => key === "overthinking_prevention_enabled" ? [key, "nsfw_asset_generation_enabled"] : [key]);
        const dependent = section.dependent?.filter(key => !emotionHiddenKeys.has(key));
        const advanced = section.advanced?.filter(key => !emotionHiddenKeys.has(key));
        return { ...section, keys, dependent: dependent?.length ? dependent : undefined,
          advanced: advanced?.length ? advanced : undefined };
      })
      .filter(section => section.keys.length || section.dependent?.length || section.advanced?.length),
  })),
  // 감정 삽화 공방 전용: 생성할 감정 에셋 개수 설정 탭.
  { id: "emotion_assets", title: "감정 에셋 개수 설정",
    description: "감정 삽화에서 생성할 감정 에셋 개수를 설정합니다.",
    sections: [{ title: "", keys: ["output_count_min", "output_count_max", "scene_mode"] }] as GenerationSection[] },
];
const emotionGenerationTabs = Object.fromEntries(emotionGenerationGroups.map(group => [group.id, group.title]));
const dependencies: Record<string, string[]> = {
  single_char_mask_skip_enabled: ["single_char_mask_enabled"],
  single_char_mask_skip_area_percent: ["single_char_mask_enabled", "single_char_mask_skip_enabled"],
  speak_emotions: ["speak_emotion_enabled"],
  call1_context_turns: ["outfit_enabled"], npc_generation_deferred: ["npc_generation_enabled"],
  call2_parallel_enabled: ["call2_plan_enabled"],
  call2_parallel_max_concurrency: ["call2_plan_enabled", "call2_parallel_enabled"],
};
for (const suffix of ["enabled", "remaining", "progress_enabled", "progress_threshold", "tps_enabled", "tps_threshold", "condition_operator"]) {
  dependencies[`call2_parallel_slow_retry_${suffix}`] = ["call2_plan_enabled", "call2_parallel_enabled",
    ...(suffix === "enabled" ? [] : ["call2_parallel_slow_retry_enabled"]),
    ...(suffix === "progress_threshold" ? ["call2_parallel_slow_retry_progress_enabled"] : []),
    ...(suffix === "tps_threshold" ? ["call2_parallel_slow_retry_tps_enabled"] : []),
  ];
}
const fieldHelp: Record<string, string> = {
  local_llm_load_after_generation: "삽화 작업이 끝나면, LLM 모델을 GPU에 올립니다.",
  image_model_load_at_detail: "detail이 시작하는 시점에서 그림 모델을 미리 GPU에 올립니다.\n초기 웜업 시간을 LLM 생성 시간에 가릴 수 있습니다.",
  illustration_output_mode: "원본 에셋 장수는 에셋 선택 탭에서 설정합니다. 만화 삽화 모드는 MOSS 그림 모델에서만 선택할 수 있습니다.",
  moss_dialogue_repair_enabled: "항상 OFF로 고정됩니다. 말풍선 대사를 교정하지 않고 그대로 그립니다.",
  nsfw: "NSFW 및 물리 접촉 지침을 추가합니다. 호환성 지침은 항상 적용됩니다.",
  overthinking_prevention_enabled: "DETAIL·DETAIL-MOSS·DETAIL-MANGA의 LLM 시스템 프롬프트 맨 앞에 과도한 추론을 줄이는 지침을 추가합니다.",
  manga_composition_hint_enabled: "만화 프롬프트의 기존 컷 구성 지침 아래에 3~5컷 배치 목록을 추가합니다.",
  manga_disable_bubbles: "만화 출력에서 말풍선·텍스트를 제외합니다.",
  nsfw_asset_generation_enabled: "캐릭터의 NSFW 복장·감정 에셋을 생성합니다.",
  profile_context_translate: "다중 프로필 캐릭터의 등록 문맥을 영어로 미리 번역해 프로필 판정에 사용합니다. 끄면 번역 LLM 호출 없이 원문을 그대로 사용합니다.",
  outfit_enabled: "복장·헤어를 미리 추적합니다. 끄면 후속 단계가 원문과 프로필에서 현재 상태를 판단합니다.",
  time_place_enabled: "시간·장소·조명을 미리 추적하고 DETAIL이 해당 조건을 이어받습니다.",
  npc_generation_enabled: "대화에 등장한 미등록 인물의 외모와 복장을 정해 캐릭터 카드와 대표 이미지를 만듭니다. 얼굴 이미지(FACE)는 그 대표 이미지를 생성할 때 추적한 얼굴 영역을 잘라 저장합니다.",
  npc_generation_deferred: "NPC 외형 예측은 그대로 먼저 진행하고, 대표 이미지 Comfy 호출만 이미지 생성 단계(STEP 6)로 미룹니다.",
  npc_lifetime: "같은 봇에서 삽화를 이 횟수만큼 연속 생성하는 동안 등장하지 않으면 자동 생성 캐릭터를 삭제합니다. 다시 등장하면 수명이 초기화됩니다. 모든 봇의 기존·신규 자동 생성 캐릭터에 적용됩니다.",
  call2_plan_enabled: "장면을 미리 계획합니다. 끄면 DETAIL이 원문에서 직접 계획합니다.",
  key_visual: "켜면 첫 DETAIL 배치에서 대표 이미지 한 장을 함께 작성합니다. 만화 삽화에서는 일반 만화 페이지 외에 KEYVIS 대표 만화 1페이지를 추가하며, 만화 이미지 크기·확대 배율·시드를 사용합니다.",
  single_char_mask_enabled: "1인 장면에도 현재 모델에서 마스크를 함께 예측합니다.",
  single_char_mask_skip_enabled: "예측한 1인 영역의 화면 점유율이 아래 기준을 초과하면 마스크를 생략합니다.",
  single_char_mask_skip_area_percent: "예측한 인물 영역이 이 비율을 초과하면 1인 마스크를 생략합니다.",
  postprocess_enabled: "대사를 생성하고 이미지에 자막·텍스트 교체를 합성합니다. 세부 옵션은 POST PROCESS에서 설정합니다.",
  original_asset_count: "원본 에셋 또는 원본 에셋을 포함한 출력 모드에 적용됩니다.",
  manga_output_count_min: "한 페이지는 1~5컷입니다.",
  output_count_min: "선택할 일반 장면 수의 하한입니다.", output_count_max: "선택할 일반 장면 수의 상한입니다. PLAN과 DETAIL은 배정된 장면 수를 유지합니다.",
  manga_output_count_max: "선택할 만화 페이지 수의 상한입니다.",
  scene_mode: "두 모드 모두 최소·최대 범위를 적용합니다. 자동 선택은 범위 안에서 문맥에 맞는 수량을 고릅니다.",
  manga_scene_mode: "최소·최대 범위 안에서 페이지 수를 정합니다.",
  character_limit: "한 이미지에 배치할 최대 캐릭터 수입니다. 봇의 이미지 정책도 함께 적용합니다.",
  manga_character_limit: "한 페이지에 등장할 최대 캐릭터 수입니다.",
  focus: "장면 선택과 시각 계획에서 우선할 캐릭터입니다. 비워 두면 대화 문맥에서 결정합니다.",
  manga_focus: "만화 사건 선택과 컷 계획에서 우선할 캐릭터입니다.",
  direction: "장면 선택과 PLAN·DETAIL의 구도·동작·접촉 판단에 전달할 연출 지시입니다.",
  manga_direction: "만화 사건 선택과 컷·대사 구성에 적용할 연출 지시입니다.",
  speak_language: "일반 장면 대사의 언어를 선택합니다. 대표 이미지는 대사 대상에서 제외합니다.",
  speak_emotion_enabled: "대사에 감정 태그를 덧붙입니다.",
  speak_emotions: "감정 이름을 세미콜론(;)으로 구분합니다.",
  prompt_format: "선택한 삽화 워크플로우에 맞는 프롬프트 구조가 자동으로 적용됩니다.",
  positive_note: "모든 장면의 최종 긍정 프롬프트에 덧붙일 내용입니다.",
  negative_note: "모든 장면의 최종 부정 프롬프트에 덧붙일 내용입니다.",
  call2_parallel_enabled: "선택 장면을 여러 PLAN 요청으로 나누어 처리합니다.",
  call2_parallel_max_concurrency: "장면을 이 개수에 균등 분배합니다. 연결된 LLM의 동시 요청 상한도 적용합니다.",
  call2_parallel_slow_retry_enabled: "일부 요청만 뒤처질 때 같은 작업을 한 번 더 요청해 먼저 검증된 결과를 사용합니다.",
  call2_parallel_slow_retry_remaining: "실행 중인 작업이 이 개수 이하로 남았을 때 느린 요청을 검사합니다.",
  call2_parallel_slow_retry_progress_threshold: "추정 진행률이 이 값보다 낮으면 재시도 조건을 만족합니다.",
  call2_parallel_slow_retry_tps_threshold: "추정 초당 토큰 생성 속도가 이 값보다 낮으면 재시도 조건을 만족합니다.",
};
type ToggleOption = [string, string, boolean?];
function ToggleField({ name, value, numeric, disabled, onChange, options: optionOverride, help: helpOverride, label: labelOverride }: { name: string; value: any; numeric: boolean; disabled: boolean; onChange: (value: any, immediate?: boolean) => void; options?: ToggleOption[]; help?: string; label?: string }) {
  const label = labelOverride || labels[name] || name;
  const options = optionOverride || choices[name] || (name.endsWith("condition_operator") ? [["and", "모두 만족"], ["or", "하나 이상 만족"]] : null);
  const strategy = ["local_llm_load_after_generation", "image_model_load_at_detail"].includes(name);
  const helpText = helpOverride ?? fieldHelp[name];
  const help = helpText && <small className="field-help">{strategy && <>토글 설명<br /></>}<span style={strategy ? { whiteSpace: "pre-line" } : undefined}>{helpText}</span>{name === "image_model_load_at_detail" && <><br /><span className="illust-strategy-warning">주의: DETAIL을 로컬 LLM으로 처리 중이라면 사용하지 마세요</span></>}</small>;
  const stacked = ["direction", "manga_direction", "positive_note", "negative_note", "prompt_format"].includes(name);
  return <div className={`illust-generation-field${stacked ? " stacked" : ""}`} data-disabled={disabled}>
    <label className="illust-generation-copy" htmlFor={`generation-${name}`}><strong>{label}</strong>{help}</label>
    <div className={`illust-generation-control${numeric && !options ? " numeric" : ""}`}>
      {options ? <select id={`generation-${name}`} name={name} aria-label={label} disabled={disabled} value={String(value)} onChange={event => onChange(typeof value === "boolean" ? event.target.value === "true" : numeric ? Number(event.target.value) : event.target.value, true)}>
        {!options.some(([key]) => key === String(value)) && <option value={String(value)}>{String(value)}</option>}{options.map(([key, text, optionDisabled]) => <option key={key} value={key} disabled={optionDisabled}>{text}</option>)}
      </select> : typeof value === "boolean" ? <span className="illust-generation-switch">
        <input id={`generation-${name}`} name={name} aria-label={label} disabled={disabled} type="checkbox" role="switch" checked={value} onChange={event => onChange(event.target.checked, true)} /><span aria-hidden="true" className="illust-generation-switch-track" /><span className="illust-generation-state" aria-hidden="true">{value ? "ON" : "OFF"}</span>
      </span> : numeric ? <input id={`generation-${name}`} name={name} aria-label={label} disabled={disabled} required type="number" min={name === "npc_lifetime" ? 1 : 0} max={name.includes("character_limit") ? 3 : undefined} step={name.includes("percent") || name.includes("tps") ? "any" : 1} value={value} onChange={event => onChange(event.target.value === "" ? "" : Number(event.target.value))} />
      : stacked ? <textarea id={`generation-${name}`} name={name} aria-label={label} disabled={disabled} rows={5} value={value ?? ""} onChange={event => onChange(event.target.value)} />
      : <input id={`generation-${name}`} name={name} aria-label={label} disabled={disabled} value={value ?? ""} onChange={event => onChange(event.target.value)} />}
    </div>
  </div>;
}

const promptTabs = { settings: "생성 설정", special_strategy: "특수 전략", logs: "생성 로그", history: "채팅 히스토리", prompts: "프롬프트 구성", flow: "노드·Think 확인" };
// 감정 삽화 공방은 표정 프리셋 설정 탭을 특수 전략 옆에 하나 더 둔다.
const { settings: settingsTab, special_strategy: specialStrategyTab, ...restPromptTabs } = promptTabs;
const emotionPromptTabs = { settings: settingsTab, special_strategy: specialStrategyTab, presets: "프리셋 설정", ...restPromptTabs };
const identityPath = (path: string) => path;
const changes = (current: IllustrationData, saved: IllustrationData | null) => Object.fromEntries(Object.entries(current).filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(saved?.[key])));
// A completed request must not replace edits made while it was in flight.
const acceptSaved = (current: IllustrationData | null, sent: IllustrationData, stored: IllustrationData) => ({ ...stored, ...changes(current || {}, sent) });

function ModelFamilyField({ value, disabled, onChange }: {
  value: "moss" | "anima" | ""; disabled: boolean; onChange: (value: string) => void;
}) {
  return <div className="illust-generation-field">
    <label className="illust-generation-copy" htmlFor="generation-illustration_model_family"><strong>그림 모델</strong>
      <small className="field-help">MOSS 또는 ANIMA 그림 모델 프리셋을 선택합니다. 변경하면 활성 그림 모델 프리셋이 전환됩니다.</small>
    </label>
    <div className="illust-generation-control">
      <select id="generation-illustration_model_family" name="illustration_model_family" aria-label="그림 모델"
        disabled={disabled} value={value} onChange={event => onChange(event.target.value)}>
        {!value && <option value="" disabled>모델 확인 중…</option>}
        <option value="moss">MOSS</option>
        <option value="anima">ANIMA</option>
      </select>
    </div>
  </div>;
}

export function IllustPrompts({ botSelected, scope = "illustration" }: { botSelected: string; scope?: "illustration" | "emotion" }) {
  const [tab, setTab] = useState("settings");
  const [generationTab, setGenerationTab] = useState("pipeline");
  const [postprocessVisited, setPostprocessVisited] = useState(false);
  const [visited, setVisited] = useState<Set<string>>(() => new Set(["settings"]));
  const [presetChars, setPresetChars] = useState(0);
  const [promptRevision, setPromptRevision] = useState(0);
  function selectTab(value: string) { setTab(value); setVisited(current => new Set([...current, value])); }
  // 감정 공방은 같은 화면을 전용 저장소로 연결한다. 토글·지침·히스토리 모두
  // scope가 붙은 요청만 사용하므로 삽화 공방 설정과 섞이지 않는다.
  const request = scope === "emotion" ? emotionIllustrationRequest : illustrationRequest;
  const scopedPath = scope === "emotion" ? emotionScopePath : identityPath;
  const scopeGenerationGroups = scope === "emotion" ? emotionGenerationGroups : generationGroups;
  const scopeGenerationTabs = scope === "emotion" ? emotionGenerationTabs : generationTabs;
  const togglesResource = useIllustrationResource(scopedPath("illustration_context/toggles"), false, request);
  const promptsResource = useIllustrationResource(scopedPath("illustration_context/prompts"), false, request);
  const configResource = useIllustrationResource(scopedPath("illustration/config"), false, request);
  // The active illustration model preset is global; this selector switches it
  // without touching the per-generation toggles saved below.
  const modelPresetResource = useIllustrationResource(scope === "illustration" ? "illustration/presets" : null, false, request);
  const activeModelPreset = (modelPresetResource.data?.presets || []).find(
    (item: IllustrationData) => item.id === modelPresetResource.data?.active_id);
  const modelFamily = modelPresetResource.data ? illustrationPresetFamily(activeModelPreset) : "";
  const [pendingModelFamily, setPendingModelFamily] = useState<"moss" | "anima" | "">("");
  const modelAction = useIllustrationAction();
  const [toggles, setToggles] = useState<IllustrationData | null>(null);
  const [prompts, setPrompts] = useState<IllustrationData | null>(null);
  const [config, setConfig] = useState<IllustrationData | null>(null);
  const promptAction = useIllustrationAction();
  const form = useRef<HTMLFormElement>(null);
  const [saveDelay, setSaveDelay] = useState(600);
  useEffect(() => { if (togglesResource.data) setToggles(togglesResource.data.toggles); }, [togglesResource.data]);
  useEffect(() => { if (promptsResource.data) setPrompts(promptsResource.data); }, [promptsResource.data]);
  useEffect(() => { if (configResource.data) setConfig(configResource.data.config); }, [configResource.data]);
  // ANIMA 활성 상태에서는 만화 출력 모드를 쓸 수 없으므로, 이전에 저장된
  // 만화 모드가 남아 있으면 일반 삽화로 되돌린다.
  useEffect(() => {
    if (scope !== "illustration" || modelFamily !== "anima") return;
    setToggles(current => current && ["manga", "manga_both"].includes(current.illustration_output_mode)
      ? { ...current, illustration_output_mode: "illustration" } : current);
  }, [modelFamily, scope]);
  const [savedToggles, setSavedToggles] = useState<IllustrationData | null>(null);
  const [savedConfig, setSavedConfig] = useState<IllustrationData | null>(null);
  const [savedPrompts, setSavedPrompts] = useState<IllustrationData | null>(null);
  useEffect(() => { if (togglesResource.data) setSavedToggles(togglesResource.data.toggles); }, [togglesResource.data]);
  useEffect(() => { if (configResource.data) setSavedConfig(configResource.data.config); }, [configResource.data]);
  useEffect(() => { if (promptsResource.data) setSavedPrompts(promptsResource.data); }, [promptsResource.data]);
  const dirty = !!toggles && !!config && !!savedToggles && !!savedConfig && (!!Object.keys(changes(toggles, savedToggles)).length || !!Object.keys(changes(config, savedConfig)).length);
  const promptDirty = !!prompts && !!Object.keys(changes(prompts, savedPrompts)).length;
  // 감정 삽화 공방도 에셋 개수 설정 탭에서 장면 수 범위를 노출하므로 함께 검증한다.
  const invalidRange = !!toggles && (Number(toggles.output_count_min) > Number(toggles.output_count_max)
    || scope !== "emotion" && Number(toggles.manga_output_count_min) > Number(toggles.manga_output_count_max));
  const autosave = useWorkflowAutosave({ label: "삽화 프롬프트 설정 · 생성 옵션", value: JSON.stringify([toggles, config]), dirty, form, delay: saveDelay,
    validation: invalidRange ? (scope === "emotion" ? "최소 장면 수는 최대 장면 수보다 클 수 없습니다." : "최소 장면·페이지 수는 최대 장면·페이지 수보다 클 수 없습니다.") : "",
    save: async () => {
      if (!toggles || !config) return;
      const sentToggles = toggles;
      const sentConfig = config;
      const configChanges = changes(sentConfig, savedConfig);
      const toggleChanges = changes(sentToggles, savedToggles);
      if (Object.keys(configChanges).length) {
        const result = await request("illustration/config", configChanges);
        setSavedConfig(result.config);
        setConfig(current => acceptSaved(current, sentConfig, result.config));
      }
      if (Object.keys(toggleChanges).length) {
        const latest = await request("illustration_context/toggles");
        const result = await request("illustration_context/toggles", { toggles: { ...latest.toggles, ...toggleChanges } });
        setSavedToggles(result.toggles);
        setToggles(current => acceptSaved(current, sentToggles, result.toggles));
      }
    },
  });
  useUnsavedChanges({ label: "삽화 프롬프트 설정 · 지침 편집", dirty: promptDirty, busy: promptAction.busy, error: promptAction.error,
    detail: "지침 저장을 눌러 변경 사항을 저장하세요.", onReview: () => selectTab("prompts") });

  if (!toggles || !prompts || !config) return <Panel title="삽화 프롬프트 설정"><IllustrationLoading error={togglesResource.error || promptsResource.error || configResource.error} retry={() => { togglesResource.reload(); promptsResource.reload(); configResource.reload(); }} /></Panel>;
  async function selectModelFamily(value: string) {
    const family = value === "moss" ? "moss" : "anima";
    const document = modelPresetResource.data;
    if (!document) return;
    const storageKey = `soya-illust-model-family:${family}`;
    const remembered = window.localStorage.getItem(storageKey);
    // Remember the preset we are leaving so switching back restores it.
    const leaving = (document.presets || []).find((item: IllustrationData) => item.id === document.active_id);
    if (leaving) window.localStorage.setItem(
      `soya-illust-model-family:${illustrationPresetFamily(leaving)}`, leaving.id);
    const presets = (document.presets || []).filter(
      (item: IllustrationData) => illustrationPresetFamily(item) === family);
    const target = presets.find((item: IllustrationData) => item.id === document.active_id)
      || presets.find((item: IllustrationData) => item.id === remembered)
      || presets[0];
    if (!target) {
      setPendingModelFamily("");
      return;
    }
    // ANIMA에서는 만화 출력 모드를 쓸 수 없으므로 일반 삽화로 되돌린다.
    if (family === "anima" && ["manga", "manga_both"].includes(toggles?.illustration_output_mode)) {
      setToggles(current => current ? { ...current, illustration_output_mode: "illustration" } : current);
    }
    window.localStorage.setItem(storageKey, target.id);
    if (target.id === document.active_id) {
      setPendingModelFamily("");
      return;
    }
    setPendingModelFamily(family);
    await modelAction.run(async () => {
      await request("illustration/presets", {
        action: "select", target_id: target.id, preset_id: document.active_id,
        revision: document.revision, workshop: "illustration",
      });
      modelPresetResource.reload();
      configResource.reload();
    }, `${target.name} 모델로 전환했습니다.`);
    setPendingModelFamily("");
  }
  function field(name: string) {
    if (name === "illustration_model_family") {
      if (scope !== "illustration") return null;
      return <ModelFamilyField key={name} value={pendingModelFamily || modelFamily}
        disabled={modelAction.busy || !modelPresetResource.data}
        onChange={value => void selectModelFamily(value)} />;
    }
    // 일반 삽화 공방은 합성 모드를 선택할 수 없으므로, 합성 모드가 저장돼
    // 있으면 출력 모드 항목 자체를 숨긴다. 감정 삽화 공방은 합성 설정 탭이
    // 없으므로 저장된 모드와 무관하게 출력 모드를 항상 노출한다.
    if (name === "illustration_output_mode" && scope !== "emotion" && compositeOutputModes.includes(toggles!.illustration_output_mode)) return null;
    const emotionOutputMode = scope === "emotion" && name === "illustration_output_mode";
    const emotionAssetCount = scope === "emotion" && name in emotionAssetLabels;
    // 만화 삽화 출력은 MOSS 그림 모델 전용이므로 ANIMA일 때는 선택할 수 없다.
    const outputModeOptions: ToggleOption[] = choices.illustration_output_mode.map(
      ([key, text]): ToggleOption => [key, text,
        (pendingModelFamily || modelFamily) === "anima" && ["manga", "manga_both"].includes(key)]);
    const retryPrefix = name.includes("_parallel_slow_retry_") ? name.split("_slow_retry_")[0] : "";
    const mossModel = (pendingModelFamily || modelFamily) === "moss";
    // MOSS는 DETAIL-MOSS가 만든 bbox 객체 레이아웃을 그대로 쓰므로 V3 영역 마스크 설정을 제어하지 않는다.
    const maskKeys = ["single_char_mask_enabled", "single_char_mask_skip_enabled", "single_char_mask_skip_area_percent"];
    const maskLocked = mossModel && maskKeys.includes(name);
    const mangaMode = ["manga", "manga_both"].includes(toggles!.illustration_output_mode);
    if (["manga_composition_hint_enabled", "manga_disable_bubbles"].includes(name) && !mangaMode) return null;
    const postprocessLocked = scope === "illustration" && mangaMode && !!toggles!.manga_disable_bubbles && name === "postprocess_enabled";
    const planLocked = mangaMode && (name === "call2_plan_enabled" || (dependencies[name] || []).includes("call2_plan_enabled"));
    const disabled = postprocessLocked || planLocked || maskLocked || name === "prompt_format" || (dependencies[name] || []).some(key => !toggles![key]) || !!retryPrefix && Number(toggles![`${retryPrefix}_max_concurrency`]) < 2
      || name === "original_asset_count" && !["original_asset", "both", "manga_both", "composite_both"].includes(toggles!.illustration_output_mode)
      || name === "moss_dialogue_repair_enabled";
    const rawValue = name === "postprocess_enabled" ? config!.postprocess_enabled : ["local_llm_load_after_generation", "image_model_load_at_detail", "nsfw_asset_generation_enabled", "overthinking_prevention_enabled", "npc_generation_deferred", "manga_disable_bubbles"].includes(name) ? toggles![name] ?? false : toggles![name];
    // Preserve the saved master switch so leaving manga releases this override.
    const value = postprocessLocked || planLocked && name === "call2_plan_enabled" || name === "moss_dialogue_repair_enabled" || maskLocked && ["single_char_mask_enabled", "single_char_mask_skip_enabled"].includes(name) ? false
      : emotionOutputMode && rawValue === "composite_both" ? "composite" : rawValue;
    return <ToggleField key={name} name={name} value={value} numeric={typeof savedToggles?.[name] === "number"} disabled={disabled}
      label={emotionAssetCount ? emotionAssetLabels[name] : undefined}
      options={emotionOutputMode ? emotionOutputChoices : name === "illustration_output_mode" ? outputModeOptions : undefined}
      help={postprocessLocked ? "말풍선 끄기가 켜진 만화 삽화에서는 POST PROCESS를 사용하지 않습니다. 일반 삽화로 전환하면 잠금이 해제됩니다." : planLocked && name === "call2_plan_enabled" ? "만화 삽화에서는 DETAIL-MANGA가 컷 구성과 대사를 직접 작성하므로 PLAN을 사용하지 않습니다." : maskLocked && name === "single_char_mask_enabled" ? "MOSS는 DETAIL-MOSS가 만든 bbox 객체 레이아웃을 그대로 사용하므로 영역 마스크를 쓰지 않습니다." : emotionOutputMode ? "감정 삽화 모드는 캐릭터의 복장과 감정 이미지를 생성합니다. 배경 합성 모드는 만들어진 캐릭터를 불러와 배경에 합성합니다." : emotionAssetCount ? emotionAssetHelp[name] : undefined}
      onChange={(value, immediate) => {
        setSaveDelay(immediate ? 0 : 600);
        if (name === "postprocess_enabled") setConfig(current => ({ ...current, postprocess_enabled: value }));
        else setToggles(current => ({ ...current, [name]: value }));
      }} />;
  }
  return <div className="illust-stack">
    <SettingsTabs id="illust-prompts" label="삽화 프롬프트 설정 메뉴"
      values={scope === "emotion" ? emotionPromptTabs : promptTabs} value={tab} onChange={selectTab} />
    {promptDirty && <p className="field-help" role="status">프롬프트 구성에 저장하지 않은 변경이 있습니다.</p>}
    <div role="tabpanel" id="illust-prompts-panel-settings" aria-labelledby="illust-prompts-tab-settings" hidden={tab !== "settings"}>
    <Panel title="생성 설정" className="illust-generation-settings" aside={<span role="status"><Badge tone={autosave.error || autosave.inputError ? "rose" : undefined}>{autosave.status}</Badge></span>}>
    <div className="illust-body illust-generation-layout">
      <SettingsTabs id="illust-generation" orientation="vertical" className="illust-generation-tabs" label="생성 설정 카테고리" values={scopeGenerationTabs} value={generationTab} onChange={value => { setGenerationTab(value); if (value === "postprocess") setPostprocessVisited(true); }} />
      <div className="illust-generation-main">
    <form ref={form} id="illust-generation-form" noValidate onSubmit={event => { event.preventDefault(); void autosave.retry(); }}
      onBlur={event => { if (event.target.matches("input:not([type=checkbox]), textarea")) void autosave.retry(); }}>
        <fieldset className="illust-form">
          <p className="field-help">변경 사항은 자동 저장됩니다. 숫자·텍스트는 입력을 멈추거나 입력칸을 벗어나면 저장됩니다.</p>
          <IllustrationFeedback error={modelAction.error || modelPresetResource.error || autosave.error || autosave.inputError} message={modelAction.message} />
          {autosave.error && <button type="button" className="button" disabled={autosave.busy || !!autosave.inputError} onClick={() => void autosave.retry()}>자동 저장 다시 시도</button>}
          {scopeGenerationGroups.map(group => <div key={group.id} role="tabpanel" id={`illust-generation-panel-${group.id}`} aria-labelledby={`illust-generation-tab-${group.id}`} data-generation-tab={group.id} hidden={generationTab !== group.id} className="illust-generation-content">
            <header className="illust-generation-heading"><h3>{group.title}</h3><p>{group.description}</p></header>
            <div className="illust-generation-sections">{group.sections.map((section, index) => <fieldset key={index} className={`illust-generation-section${section.title ? "" : " plain"}${section.paired ? " paired" : ""}`} disabled={!!section.mode && section.mode !== (["manga", "manga_both"].includes(toggles.illustration_output_mode) ? "manga" : "illustration")}>
              {section.title && <legend>{section.title}</legend>}
              <div className="illust-fields">{section.keys.map(field)}</div>
              {section.dependent && <div className="illust-generation-dependent"><div className="illust-fields">{section.dependent.map(field)}</div></div>}
              {section.advanced && <div className="illust-generation-advanced"><h4>{section.advancedTitle}</h4><div className="illust-generation-dependent"><div className="illust-fields">{section.advanced.map(field)}</div></div></div>}
            </fieldset>)}</div>
          </div>)}
        </fieldset>
    </form>
    <div className="illust-generation-postprocess" hidden={generationTab !== "postprocess"}>{postprocessVisited && <IllustPostprocess botName={botSelected} outputMode={toggles.illustration_output_mode} active={tab === "settings" && generationTab === "postprocess"} />}</div>
      </div>
    </div>
    </Panel>
    </div>
    <div role="tabpanel" id="illust-prompts-panel-special_strategy" aria-labelledby="illust-prompts-tab-special_strategy" hidden={tab !== "special_strategy"}>
      <Panel title="특수 전략" aside={<span role="status"><Badge tone={autosave.error || autosave.inputError ? "rose" : undefined}>{autosave.status}</Badge></span>}>
        <div className="illust-body">
          <p className="field-help">변경 사항은 자동 저장됩니다.</p>
          <IllustrationFeedback error={autosave.error || autosave.inputError} />
          {autosave.error && <button type="button" className="button" disabled={autosave.busy || !!autosave.inputError} onClick={() => void autosave.retry()}>자동 저장 다시 시도</button>}
          <fieldset className="illust-generation-section">
            <legend>로컬 LLM 대응 전략</legend>
            <div className="illust-fields">{["local_llm_load_after_generation", "image_model_load_at_detail"].map(field)}</div>
          </fieldset>
          <fieldset className="illust-generation-section">
            <legend>NPC 생성</legend>
            <div className="illust-fields">{field("npc_generation_deferred")}</div>
          </fieldset>
        </div>
      </Panel>
    </div>
    <div role="tabpanel" id="illust-prompts-panel-prompts" aria-labelledby="illust-prompts-tab-prompts" hidden={tab !== "prompts"}><div className="illust-stack">
    {visited.has("prompts") && <IllustPromptStructure toggles={toggles} active={tab === "prompts"} revision={promptRevision} request={request} />}
    <IllustPromptEditor prompts={prompts} savedPrompts={savedPrompts} scope={scope} {...promptAction} onChange={(key, value) => setPrompts(current => ({ ...current, [key]: value }))}
      onSave={() => void promptAction.run(async () => {
        const edited = changes(prompts, savedPrompts);
        const result = await request("illustration_context/prompts", edited);
        const saved = new Set<string>(Array.isArray(result.saved) ? result.saved : []);
        const accepted = Object.fromEntries(Object.entries(edited).filter(([key]) => saved.has(key)));
        setSavedPrompts(current => ({ ...current, ...accepted }));
        if (Object.keys(accepted).length) setPromptRevision(value => value + 1);
        const missing = Object.keys(edited).filter(key => !saved.has(key));
        if (missing.length) throw new Error(`저장되지 않은 지침: ${missing.join(", ")}. 변수명과 서버 응답을 확인하세요.`);
      }, "지침을 저장했습니다. 새 호출부터 적용됩니다.")}
      onReload={() => void promptAction.run(async () => {
        const data = await request("illustration_context/prompts");
        setPrompts(data); setSavedPrompts(data); setPromptRevision(value => value + 1);
      }, "저장된 지침을 다시 불러왔습니다.")} />
    </div></div>
    <div role="tabpanel" id="illust-prompts-panel-presets" aria-labelledby="illust-prompts-tab-presets" hidden={tab !== "presets"}>
      {scope === "emotion" && visited.has("presets") && <Panel title="프리셋 설정"
        aside={<span className="illust-emotion-preset-header-aside">
          <span className={`illust-emotion-preset-charcount${presetChars > EMOTION_PRESET_CHAR_LIMIT ? " over" : ""}`}
            role="status" aria-live="polite" title="SFW·NSFW 전체 감정 이름과 표정 태그 글자수">
            {presetChars.toLocaleString("ko-KR")} / {EMOTION_PRESET_CHAR_LIMIT.toLocaleString("ko-KR")}자
          </span>
          <span role="status"><Badge tone={autosave.error || autosave.inputError ? "rose" : undefined}>{autosave.status}</Badge></span>
        </span>}>
        <EmotionPresetSettings toggles={toggles} onChange={patch => {
          setSaveDelay(600);
          setToggles(current => ({ ...current, ...patch }));
        }} onTotalCharsChange={setPresetChars} />
      </Panel>}
    </div>
    <div role="tabpanel" id="illust-prompts-panel-logs" aria-labelledby="illust-prompts-tab-logs" hidden={tab !== "logs"}>{visited.has("logs") && <IllustGenerationLogs active={tab === "logs"} />}</div>
    <div role="tabpanel" id="illust-prompts-panel-history" aria-labelledby="illust-prompts-tab-history" hidden={tab !== "history"}>{visited.has("history") && <IllustHistory active={tab === "history"} />}</div>
    <div role="tabpanel" id="illust-prompts-panel-flow" aria-labelledby="illust-prompts-tab-flow" hidden={tab !== "flow"}>{visited.has("flow") && <IllustNodeFlow toggles={toggles} botSelected={botSelected} active={tab === "flow"} request={request} scope={scope} />}</div>
  </div>;
}
