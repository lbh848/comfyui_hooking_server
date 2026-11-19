import { useState } from "react";
import { RefreshCw, Save } from "lucide-react";
import { Badge, Modal, SettingsTabs } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import type { IllustrationData } from "./illustration";

type PromptGroup = { id: string; title: string; description: string; fields: { key: string; label: string; help?: string }[] };

// These are presentation groups. Field keys remain the canonical keys returned
// by illustration_context/prompts and consumed by PROMPT_FILES in the engine.
const promptGroups: PromptGroup[] = [
  { id: "character", title: "CHARACTER-RESOLVE", description: "현재 본문의 등장인물과 삽화 장면을 함께 선택합니다. 두 지침과 공유 프롬프트를 함께 사용합니다.", fields: [
    { key: "character_resolve", label: "등장인물 판단 지침" },
    { key: "call2_select", label: "통합 장면 선택 지침" },
  ] },
  { id: "manga_character", title: "MANGA-CHARACTER-RESOLVE", description: "현재 본문의 등장인물과 만화로 요약할 사건을 선택합니다. 등장인물 판단 지침은 CHARACTER-RESOLVE에서, 공용 지침은 공유 프롬프트에서 편집합니다.", fields: [
    { key: "manga_select", label: "만화 사건 선택 지침" },
  ] },
  { id: "profile", title: "PROFILE-RESOLVE", description: "확정 등장인물의 프로필과 적용 시점을 결정합니다.", fields: [
    { key: "profile_resolve", label: "프로필 선택 지침" },
  ] },
  { id: "emotion_resolve", title: "OUTFIT-RESOLVE · CURRENT 복장 결정", description: "캐릭터·감정·프로필이 확정된 선택 장면마다 입을 복장 하나를 정합니다. 감정 삽화 공방에서만 사용합니다.", fields: [
    { key: "emotion_resolve", label: "장면별 복장 결정 지침" },
  ] },
  { id: "emotion_asset", title: "EMOTION-ASSET", description: "선택 장면마다 SFW/NSFW 등급과 가장 가까운 감정 프리셋을 고르고, 프로필 카드에 없는 복장 태그를 새로 정합니다. 감정 삽화 공방에서만 사용합니다.", fields: [
    { key: "emotion_asset", label: "장면별 감정 에셋 결정 지침" },
  ] },
  { id: "background_card", title: "배경 카드 예측", description: "자동 배경 모드에서 원문과 세계관 설정으로 새 배경 카드의 이미지 프롬프트를 작성합니다. 기존 배경 카드는 재사용합니다.", fields: [
    { key: "background_card", label: "배경 카드 예측 지침" },
  ] },
  { id: "outfit", title: "OUTFIT", description: "과거 문맥과 저장 이력에서 시작 복장·머리를 복원하고, 현재 원문의 변화를 추적합니다.", fields: [
    { key: "call1_enhance", label: "복장·헤어스타일 상태 추적 지침" },
  ] },
  { id: "time_place", title: "TIME & PLACE", description: "캐릭터별 시간대·장소·조명을 추적합니다. time, place, lighting은 변경된 SLOT에서 갱신하고 이전 값을 이어받습니다.", fields: [
    { key: "time_place", label: "시간·장소·조명 추적 지침" },
  ] },
  { id: "npc_generation", title: "NPC 생성 노드", description: "대화에 등장한 미등록 인물의 외모와 복장을 정해 캐릭터 카드와 대표 이미지를 만듭니다. ANIMA는 공용 지침을, MOSS는 bbox 없이 요약 캡션으로 그리는 MOSS 전용 지침을 사용합니다. 자동 삭제까지의 미사용 횟수는 생성 설정 > 파이프라인 > NPC 생성에서 정합니다.", fields: [
    { key: "npc_generation", label: "NPC 캐릭터 카드 생성 지침 (ANIMA)" },
    { key: "npc_generation_moss", label: "NPC 캐릭터 카드 생성 지침 (MOSS · bbox 없음)" },
    { key: "npc_generation_secondary", label: "2차봇 원작 캐릭터 이름 태그 지침" },
  ] },
  { id: "plan", title: "PLAN", description: "선택된 장면의 구도·동작·접촉을 계획합니다. 공용 지침은 공유 프롬프트에서 편집합니다.", fields: [
    { key: "call2_plan", label: "선택 장면 구체화 지침" },
  ] },
  { id: "detail", title: "DETAIL", description: "선택된 장면을 이미지 태그로 작성합니다. KEYVIS 통합 예측에서는 대표 이미지 지침을, 마스크 통합 예측에서는 인물 영역 배치 지침을 함께 사용합니다.", fields: [
    { key: "call2_detail", label: "이미지 태그 작성 지침" },
  ] },
  { id: "detail_moss", title: "DETAIL-MOSS", description: "MOSS 그림 모델에서 장면마다 이미지 크기를 입력받아 abs_centralized/yxyx 절대 bbox 레이아웃을 작성합니다. 이미지 중심이 (0,0)이고 y는 아래로, x는 오른쪽으로 증가합니다. 캐릭터뿐 아니라 위치가 중요한 사물도 bbox화합니다.", fields: [
    { key: "detail_moss", label: "MOSS 레이아웃 작성 지침" },
  ] },
  { id: "detail_manga", title: "DETAIL-MANGA · MOSS", description: "MOSS 그림 모델에서 만화 페이지 한 장을 page(hierarchy/flow/boundaries) + 1~5컷 panel inventory + 선택적 cross-panel actors로 작성합니다. 모델이 모든 bbox를 절대 yxyx로 계산하고, 서버가 패널·캐릭터·사물·말풍선 item으로 평면화합니다.", fields: [
    { key: "detail_manga", label: "MOSS 만화 페이지 작성 지침" },
  ] },
  { id: "keyvis", title: "KEYVIS", description: "대표 이미지 지침·작업 지침·공통 묘사 규칙을 각각 편집합니다. 외부 LLM 호환성 지침과 NSFW 및 물리 접촉 지침은 공유 프롬프트에서 편집합니다.", fields: [
    { key: "call2_keyvis", label: "대표 이미지 지침" },
    { key: "call2_keyvis_manga", label: "대표 만화 지침 (DETAIL-MANGA 통합)" },
    { key: "call2_job", label: "KEYVIS 작업 지침" },
    { key: "call2_common", label: "KEYVIS 공통 묘사 규칙" },
  ] },
  { id: "postprocess", title: "POST PROCESS", description: "POST PROCESS 모드에 맞는 대사·말풍선·자막 지침을 편집합니다. NSFW 보강 지침은 만화 대사 모드에서 NSFW를 켰을 때 함께 사용합니다.", fields: [
    { key: "call3_speak", label: "장면별 대사 지침" },
    { key: "call3_manga", label: "만화 대사·말풍선 지침" },
    { key: "call3_subtitle", label: "애니 자막 대사 지침" },
    { key: "call3_manga_nsfw", label: "만화 대사 NSFW 보강 지침" },
  ] },
  { id: "mask", title: "MULTI-CHAR-MASK", description: "인물별 영역과 공통 배경·구도를 정합니다. DETAIL·KEYVIS 지침과 함께 조립하여 현재 모델에서 마스크를 예측합니다.", fields: [
    { key: "multi_char_mask", label: "인물 영역 배치 지침" },
  ] },
  { id: "visual_correction", title: "비주얼 교정기", description: "생성 이미지의 PASS 여부와 필요한 프롬프트 교정을 판단합니다.", fields: [
    { key: "visual_correction", label: "비주얼 교정 판단·수정 지침" },
  ] },
  { id: "visual_selection", title: "결과 선택기", description: "원본과 교정본을 비교해 최종 이미지를 선택합니다.", fields: [
    { key: "visual_selection", label: "원본·교정본 비교 선택 지침" },
  ] },
  { id: "shared", title: "공유 프롬프트", description: "여러 단계에서 함께 사용하는 지침입니다. 장르 지침은 장르 입력 시, NSFW 및 물리 접촉 지침은 NSFW를 켰을 때 포함됩니다.", fields: [
    { key: "genre", label: "장르·세계관 공통 지침" },
    { key: "genre_secondary", label: "2차봇 작품 배경 지침", help: "장르가 2차봇일 때 작품명과 함께 전달됩니다. LLM의 기존 작품 지식을 배경·시대 해석에 사용합니다." },
    { key: "call2_jailbreak", label: "외부 LLM 호환성 지침" },
    { key: "call2_explicit", label: "NSFW 및 물리 접촉 지침" },
  ] },
  { id: "template", title: "출력 템플릿", description: "단계별 결과를 [SPEAK]·[Name]·[SETUP]·[CHAR]·[SUPPLEMENT]로 조립할 때 사용하는 템플릿입니다.", fields: [
    { key: "call2_preset", label: "이미지 출력 템플릿" },
  ] },
  { id: "legacy", title: "보관용", description: "프리필과 기존 ENQUEUE 경로의 지침을 편집합니다. 위 단계별 삽화 지침과 구분해 관리합니다.", fields: [
    { key: "call2_prefill", label: "보관용 프리필", help: "현재 단계별 삽화 생성에는 포함되지 않습니다." },
    { key: "call2_fallback", label: "ENQUEUE 폴백 지침", help: "기존 ENQUEUE 경로에서 응답을 작성할 때 사용합니다." },
    { key: "call2_format", label: "ENQUEUE 응답 형식", help: "기존 ENQUEUE 경로의 응답 출력 형식을 정합니다." },
  ] },
];
const groupedKeys = new Set(promptGroups.flatMap(group => group.fields.map(field => field.key)));

export function IllustPromptEditor({ prompts, savedPrompts, busy, error, message, onChange, onSave, onReload, scope = "illustration" }: {
  prompts: IllustrationData; savedPrompts: IllustrationData | null; busy: boolean; error: string; message: string;
  onChange: (key: string, value: string) => void; onSave: () => void; onReload: () => void; scope?: "illustration" | "emotion";
}) {
  const [selected, setSelected] = useState("character");
  const [confirmReload, setConfirmReload] = useState(false);
  // DETAIL 합성 삽화 is edited from the emotion illustration workshop.
  const extraKeys = Object.keys(prompts).filter(key => !groupedKeys.has(key) && key !== "composite_detail");
  // 감정 삽화 공방은 OUTFIT·PLAN·KEYVIS·MULTI-CHAR-MASK와 대사 후처리를 사용하지 않는다.
  // 일반 삽화 공방은 감정 삽화 전용 OUTFIT-RESOLVE를 사용하지 않는다.
  const hiddenGroups = scope === "emotion"
    ? new Set(["outfit", "plan", "keyvis", "mask", "postprocess", "visual_correction", "visual_selection"])
    : new Set(["emotion_resolve", "emotion_asset", "background_card", "mask", "visual_correction", "visual_selection"]);
  const groups = promptGroups.filter(group => !hiddenGroups?.has(group.id))
    .map(group => ({
      ...group,
      description: scope === "emotion" && group.id === "detail" ? "선택된 장면을 이미지 태그로 작성합니다." : group.description,
      fields: group.fields.filter(field => Object.hasOwn(prompts, field.key)),
    })).filter(group => group.fields.length);
  if (extraKeys.length) groups.push({ id: "additional", title: "기타 지침", description: "추가로 등록된 지침을 실제 변수명으로 편집합니다.", fields: extraKeys.map(key => ({ key, label: key })) });
  const activeGroup = groups.some(group => group.id === selected) ? selected : groups[0]?.id;
  const edited = Object.keys(prompts).filter(key => prompts[key] !== savedPrompts?.[key]);
  return <div id="illust-instruction-editor"><details className="panel illust-instruction-editor">
    <summary className="illust-instruction-toggle"><h2>지침 편집</h2>
    {!!edited.length && <Badge tone="rose">저장하지 않은 변경 {edited.length}개</Badge>}
    {error && <Badge tone="rose">저장 실패</Badge>}
    </summary>
    <fieldset className="illust-body illust-form" disabled={busy}>
    <div className="illust-actions">
    <button type="button" className="button" disabled={busy} onClick={() => edited.length ? setConfirmReload(true) : onReload()}><RefreshCw size={14} />지침 다시 불러오기</button>
    <button type="button" className="button primary" disabled={busy || !edited.length} onClick={onSave}><Save size={14} />{busy ? "처리 중…" : "지침 저장"}</button>
    </div>
    <p className="field-help">그룹별로 고정 지침을 편집합니다. 각 변수명은 실제 지침을 불러오고 저장하는 키이며, 저장 후 새 호출부터 적용됩니다.</p>
    <IllustrationFeedback error={error} message={message} />
    <SettingsTabs id="illust-instruction" className="illust-instruction-tabs" label="지침 편집 그룹" values={Object.fromEntries(groups.map(group => [group.id, group.title]))} value={activeGroup} onChange={setSelected} />
    {groups.map(group => <section key={group.id} role="tabpanel" id={`illust-instruction-panel-${group.id}`} aria-labelledby={`illust-instruction-tab-${group.id}`} hidden={activeGroup !== group.id} className="illust-instruction-group">
      <header><h3>{group.title}</h3><p className="field-help">{group.description}</p></header>
      {group.fields.map(({ key, label, help }, index) => <details className="illust-instruction-field" key={key} open={index === 0 || group.id === "character"}>
        <summary><code>{key}</code><span>{label}</span>{edited.includes(key) && <Badge tone="rose">변경됨</Badge>}</summary>
        <div className="illust-instruction-field-body">{help && <p className="field-help">{help}</p>}<label className="sr-only" htmlFor={`illust-prompt-value-${key}`}>{key} · {label}</label><textarea id={`illust-prompt-value-${key}`} name={key} data-prompt-key={key} className="illust-prompt-editor" rows={12} spellCheck={false} value={prompts[key]} onChange={event => onChange(key, event.target.value)} /></div>
      </details>)}
    </section>)}
  </fieldset></details>{confirmReload && <Modal title="지침 다시 불러오기" backgroundable={false} onClose={() => setConfirmReload(false)}><p>저장하지 않은 지침 변경 {edited.length}개를 버리고 저장된 지침을 다시 불러올까요?</p><div className="modal-actions"><button className="button" onClick={() => setConfirmReload(false)}>계속 편집</button><button className="button danger" onClick={() => { setConfirmReload(false); onReload(); }}>다시 불러오기</button></div></Modal>}</div>;
}
