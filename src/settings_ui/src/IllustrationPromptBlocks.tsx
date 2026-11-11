import type { IllustrationData } from "./illustration";

type PromptPath = (string | number)[];
type PromptBlock = { label: string; path: PromptPath; value: string; readOnly?: boolean };
export type PromptCategory = "generated" | "preset" | "assembled";
type PromptGroup = { category: PromptCategory; label: string; help?: string; blocks: PromptBlock[] };

const categoryHelp: Record<PromptCategory, string> = {
  generated: "장면에 맞춰 생성·구성된 배경, 구도, 캐릭터와 표정 태그입니다. 캐릭터 태그에는 프로필에서 가져온 외모가 포함될 수 있습니다.",
  preset: "사용자가 미리 지정한 품질·아티스트·부정 프롬프트, LoRA 트리거, 얼굴·눈 태그입니다. 이 그림에 저장된 값을 수정하며 원래 프리셋이나 캐릭터 카드는 바꾸지 않습니다.",
  assembled: "생성 태그와 사전 태그가 합쳐진 전체 장면·영역 공통 프롬프트입니다. 출처가 따로 저장되지 않은 태그는 이곳에서 편집합니다.",
};

// MOSS transports only a summary plus character/object (and text) items, so the
// tag editor mirrors that document instead of projecting V3 control blocks.
const mossCategoryHelp: Record<PromptCategory, string> = {
  generated: "MOSS 레이아웃의 서머리·캐릭터·오브젝트 태그입니다. 수정한 값은 투영 JSON과 다시 그리기에 바로 반영됩니다.",
  preset: "MOSS 워크플로 컨트롤러의 부정 프롬프트와 적용된 LoRA 트리거입니다. 부정 프롬프트는 여기서 수정할 수 있습니다.",
  assembled: "생성에 사용되는 서머리·캐릭터·오브젝트를 MOSS 순서대로 모은 참고 표시입니다. 수정은 생성 태그에서 합니다.",
};
const mossItemLabels: Record<string, string> = { character: "캐릭터", object: "오브젝트", text: "텍스트" };

const isMossDocument = (document: IllustrationData) =>
  !!document && typeof document.moss === "object" && !!document.moss
  && typeof document.workflow === "object" && !!document.workflow;

function mossPromptGroups(document: IllustrationData): PromptGroup[] {
  const groups: PromptGroup[] = [];
  const moss = document.moss as IllustrationData;
  const workflow = document.workflow as IllustrationData;
  const items: IllustrationData[] = Array.isArray(moss.items) ? moss.items : [];
  function group(category: PromptCategory, label: string, help?: string) {
    const result: PromptGroup = { category, label, help, blocks: [] };
    groups.push(result);
    return result;
  }
  function add(target: PromptGroup, label: string, path: PromptPath, readOnly = false) {
    const value = path.reduce<any>((node, key) => node?.[key], document);
    if (typeof value === "string") target.blocks.push({ label, path, value, readOnly });
  }
  function addLayout(category: PromptCategory, readOnly: boolean) {
    const summary = group(category, "서머리", "MOSS 레이아웃의 장면 요약입니다.");
    add(summary, "장면 요약", ["moss", "summary", "caption"], readOnly);
    const byLabel = new Map<string, number[]>();
    items.forEach((item, index) => {
      if (!item || typeof item !== "object") return;
      const label = String(item.label || "character");
      byLabel.set(label, [...(byLabel.get(label) || []), index]);
    });
    for (const [label, indices] of byLabel) {
      const title = mossItemLabels[label] || label;
      const target = group(category, title, label === "character" ? "캐릭터별 프롬프트입니다."
        : label === "object" ? "위치가 중요한 사물 프롬프트입니다." : "이미지에 넣을 텍스트 항목입니다.");
      indices.forEach((index, position) => add(target, `${title} ${position + 1}`, ["moss", "items", index, "caption"], readOnly));
    }
  }
  addLayout("generated", false);
  addLayout("assembled", true);
  const negative = group("preset", "부정 프롬프트", "MOSS 컨트롤러의 부정 프롬프트입니다. 다시 그리기에 그대로 적용됩니다.");
  add(negative, "부정 프롬프트", ["workflow", "negative"]);
  const loras = Array.isArray(workflow.loras?.list) ? workflow.loras.list : [];
  const triggers = group("preset", "LoRA 트리거 원문", "MOSS 컨트롤러에서 적용된 LoRA의 참고 원문입니다.");
  loras.forEach((entry: IllustrationData, index: number) => {
    if (typeof entry?.trigger !== "string" || !entry.trigger) return;
    const name = typeof entry.lora_path === "string" ? entry.lora_path.split(/[\\/]/).pop() : "";
    add(triggers, name || `LoRA ${index + 1}`, ["workflow", "loras", "list", index, "trigger"], true);
  });
  return groups.filter(target => target.blocks.length);
}

// Edit the stored inputs directly. Do not try to infer SETUP/CHAR boundaries
// from a combined prompt, or rebuild prompts with a second frontend builder.
function promptGroups(document: IllustrationData): PromptGroup[] {
  if (isMossDocument(document)) return mossPromptGroups(document);
  const groups: PromptGroup[] = [];
  const root = document.schema === "soya.illustration.v2" ? ["fields"] : [];
  const fields = root.length ? document.fields : document;
  function group(category: PromptCategory, label: string, help?: string) {
    const result: PromptGroup = { category, label, help, blocks: [] };
    groups.push(result);
    return result;
  }
  function add(target: PromptGroup, label: string, path: PromptPath, readOnly = false) {
    const value = path.reduce<any>((node, key) => node?.[key], fields);
    if (typeof value === "string") target.blocks.push({ label, path: [...root, ...path], value, readOnly });
  }
  function strings(target: PromptGroup, label: string, path: PromptPath) {
    const values = path.reduce<any>((node, key) => node?.[key], fields);
    if (Array.isArray(values)) values.forEach((_, index) => add(target, `${label} ${index + 1}`, [...path, index]));
  }

  const multi = fields.MULTI_CHAR;
  if (multi && typeof multi === "object" && !Array.isArray(multi)) {
    const shared = group("generated", "배경 · 구도", multi.enable === true
      ? "마스크 영역별 프롬프트를 조립할 때 사용하는 태그 조각입니다."
      : "현재 영역별 프롬프트는 비활성화되어 있습니다.");
    add(shared, "배경", ["MULTI_CHAR", "background_prompt"]);
    add(shared, "구도 · 보충 설명", ["MULTI_CHAR", "composition_prompt"]);
    const triggers = group("preset", "영역별 LoRA 트리거");
    strings(triggers, "배경 트리거", ["MULTI_CHAR", "background_trigger_list"]);
    if (Array.isArray(multi.char_inform)) multi.char_inform.forEach((_: unknown, index: number) => {
      const name = multi.char_name_list?.[index];
      const label = `캐릭터 ${index + 1}${typeof name === "string" && name ? ` · ${name}` : ""}`;
      const character = group("generated", label);
      add(character, "캐릭터 태그", ["MULTI_CHAR", "char_inform", index]);
      strings(triggers, `${label} · 트리거`, ["MULTI_CHAR", "char_trigger_list", index]);
    });
    const sharedTags = group("assembled", "영역 공통 태그", "캐릭터 태그의 앞·뒤에 붙는 조각입니다. 배경·품질·아티스트 블록과 같은 값으로 저장된 조각은 해당 블록을 수정하면 함께 반영됩니다.");
    strings(sharedTags, "캐릭터 앞", ["MULTI_CHAR", "shared_tag", "before_char"]);
    strings(sharedTags, "캐릭터 뒤", ["MULTI_CHAR", "shared_tag", "after_char"]);
  }

  const scene = group("generated", "장면 태그");
  for (const [key, label] of [["SETUP", "배경 · SETUP"], ["CHAR", "캐릭터 · CHAR"], ["SUPPLEMENT", "보충 설명 · SUPPLEMENT"]]) add(scene, label, [key]);
  const combined = group("assembled", "전체 프롬프트");
  for (const [key, label] of [["positive", "긍정 프롬프트"], ["ILXL", "ILXL"], ["UPSCALE", "업스케일"]]) add(combined, label, [key]);
  for (const model of ["ANIMA", "SDXL"]) {
    const target = group("assembled", `${model} 프롬프트`, "전체 장면에 합쳐진 태그입니다. 영역별 태그와는 별도로 적용됩니다.");
    add(target, "전체 장면", [`${model}_CONTENT`]);
    if (model === "SDXL") add(target, "전체 장면 · SDXL", ["SDXL"]);
    const preset = group("preset", `${model} 사전 태그`, "품질·아티스트 프리셋에서 붙인 블록입니다. 전체 장면에 이미 합쳐진 태그는 조립된 프롬프트에서 수정합니다.");
    add(preset, "품질", [`${model}_QUALITY`]);
    add(preset, "아티스트", [`${model}_ARTIST`]);
  }
  const negative = group("preset", "부정 프롬프트", "선택한 부정 프리셋과 추가 부정 입력을 반영한 값입니다.");
  for (const [key, label] of [["NEGATIVE_ANIMA", "ANIMA 부정"], ["NEGATIVE_SDXL", "SDXL 부정"]]) add(negative, label, [key]);
  for (const [key, label] of [["LORA_DATA", "캐릭터 LoRA"], ["FACE_LORA_DATA", "얼굴 LoRA"], ["STYLE_LORA_DATA", "스타일 LoRA"]]) {
    if (!Array.isArray(fields[key]?.list)) continue;
    const target = group("preset", `${label} 트리거 원문`, "사전 설정에서 가져온 참고용 원문입니다. 실제 적용 태그는 영역별 LoRA 트리거 또는 조립된 프롬프트에서 수정하세요.");
    fields[key].list.forEach((entry: IllustrationData, index: number) => {
      if (typeof entry?.trigger !== "string" || !entry.trigger) return;
      const name = typeof entry.lora_path === "string" ? entry.lora_path.split(/[\\/]/).pop() : "";
      add(target, name || `LoRA ${index + 1}`, [key, "list", index, "trigger"], true);
    });
  }
  // Unstructured backups do not retain whether the negative came from a
  // preset, a generator or a later manual edit.
  add(combined, "부정 프롬프트", ["negative"]);

  const faces = fields.CHAR_FACE_TAG_INFORM?.list;
  if (Array.isArray(faces)) faces.forEach((face: IllustrationData, index: number) => {
    const name = `${index + 1}${typeof face?.CHAR === "string" ? ` · ${face.CHAR}` : ""}`;
    const target = group("preset", `얼굴 · 눈 ${name}`, "캐릭터 카드의 얼굴·눈 태그와 얼굴 LoRA 트리거입니다.");
    for (const [key, label] of [["FACE_TAGS", "얼굴 태그"], ["EYE_TAGS", "눈 태그"],
      ["TRIGGER_ANIMA", "ANIMA 트리거"], ["TRIGGER_SDXL", "SDXL 트리거"]]) add(target, label, ["CHAR_FACE_TAG_INFORM", "list", index, key]);
    add(group("generated", `얼굴 표현 ${name}`, "생성된 캐릭터 태그에서 얼굴 처리에 사용할 내용을 추린 값입니다."), "표정 · 상황 태그", ["CHAR_FACE_TAG_INFORM", "list", index, "POSITIVE"]);
  });
  return groups.filter(target => target.blocks.length);
}

function updatePromptBlock(document: IllustrationData, block: PromptBlock, value: string): IllustrationData {
  // Clone only the edited branch; embedded images and unrelated settings keep
  // their original values and types, including arrays and numeric controls.
  function update(node: any, path: PromptPath): any {
    if (!path.length) return value;
    const [key, ...rest] = path;
    const result = Array.isArray(node) ? [...node] : { ...node };
    result[key] = update(node[key], rest);
    return result;
  }
  const result = update(document, block.path);
  if (["fields.MULTI_CHAR.background_prompt", "fields.ANIMA_QUALITY", "fields.ANIMA_ARTIST"].includes(block.path.join("."))) {
    const multi = result.fields.MULTI_CHAR;
    const shared = multi?.shared_tag;
    // Keep the saved copy of this fragment in its existing position. Do not
    // invent a new assembly order or replace coincidental substrings in tags.
    if (Array.isArray(shared?.before_char)) result.fields.MULTI_CHAR = {
      ...multi, shared_tag: { ...shared, before_char: shared.before_char.map((part: unknown) => part === block.value ? value : part) },
    };
  }
  return result;
}

export function IllustrationPromptBlocks({ document, category, disabled, readOnly = false, onChange }: {
  document: IllustrationData; category: PromptCategory; disabled: boolean; readOnly?: boolean; onChange: (document: IllustrationData) => void;
}) {
  const groups = promptGroups(document).filter(group => group.category === category);
  const moss = isMossDocument(document);
  return <div className="illust-prompt-blocks">
    <p className="field-help">{moss ? mossCategoryHelp[category] : readOnly ? "이 백업이 실제로 사용한 태그의 참고 표시입니다." : `${categoryHelp[category]} 수정한 값은 투영 JSON에 바로 반영됩니다.`}</p>
    {!groups.length && <p className="field-help">이 종류의 태그가 따로 저장되어 있지 않습니다. 조립된 프롬프트 또는 투영 JSON에서 확인하세요.</p>}
    {groups.map(group => <section className="illust-prompt-block-group" key={group.label}>
      <header><h3>{group.label}</h3><span>{group.blocks.length}개 블록</span></header>
      {group.help && <p className="field-help">{group.help}</p>}
      {group.blocks.map(block => <label className="field illust-prompt-block" key={block.path.join(".")}>
        <span>{block.label}{block.readOnly && <small className="illust-prompt-reference">참고 원문</small>}</span>
        <textarea aria-label={`${group.label} · ${block.label}`} rows={block.value.length > 240 ? 5 : 3} spellCheck={false}
          readOnly={disabled || readOnly || block.readOnly} value={block.value} onChange={event => onChange(updatePromptBlock(document, block, event.target.value))} />
      </label>)}
    </section>)}
  </div>;
}
