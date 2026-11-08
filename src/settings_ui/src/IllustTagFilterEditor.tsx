import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, Plus, Save, Trash2 } from "lucide-react";
import { Badge, Modal } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { illustrationRequest, query, useIllustrationAction, type IllustrationData } from "./illustration";

export type TagFilterStep = { action: string; pattern?: string; separator?: string; take?: string | number; from?: string; to?: string };
export type TagFilterProfiles = { profiles: Record<string, TagFilterStep[]>; active_profile: string };
const filterActions: Record<string, string> = {
  remove_match: "패턴 일치 제거", split_by: "구분자로 분리", replace: "치환",
  strip_prefix: "접두사 제거", strip_suffix: "접미사 제거", strip: "공백 제거", lower: "소문자화",
};

export function tagFilterStepLabel(step: TagFilterStep) {
  const label = filterActions[step.action] || step.action;
  if (step.action === "split_by") return `${label}: ${step.separator ?? "_"} · ${step.take ?? 0}`;
  if (step.action === "replace") return `${label}: ${step.from || ""} → ${step.to || "(삭제)"}`;
  return step.pattern ? `${label}: ${step.pattern}` : label;
}

export function IllustTagFilterEditor({ botName, characters, target, filters, profile, steps, disabled, onSelect, onChange, onSaved, onBusyChange }: {
  botName: string; characters: string[]; filters: TagFilterProfiles; profile: string; steps: TagFilterStep[]; disabled: boolean;
  target: "representative" | "face";
  onSelect: (name: string) => void; onChange: (steps: TagFilterStep[]) => void;
  onSaved: (filters: TagFilterProfiles, profile: string) => void; onBusyChange: (busy: boolean) => void;
}) {
  const action = useIllustrationAction();
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [preview, setPreview] = useState<IllustrationData[] | null>(null);
  const dirty = JSON.stringify(steps) !== JSON.stringify(filters.profiles[profile] || []);
  const busy = disabled || action.busy;
  const names = Object.keys(filters.profiles).sort((a, b) => a.localeCompare(b, "ko", { numeric: true }));
  const invalid = steps.some(step => step.action === "split_by" ? !step.separator : step.action === "replace" ? !step.from : ["remove_match", "strip_prefix", "strip_suffix"].includes(step.action) && !step.pattern?.trim());
  useEffect(() => { onBusyChange(action.busy); }, [action.busy, onBusyChange]);
  useEffect(() => { setPreview(null); }, [steps, characters, botName, profile, target]);

  function edit(index: number, patch: Partial<TagFilterStep>) {
    onChange(steps.map((step, i) => i === index ? { ...step, ...patch } : step));
  }
  function move(index: number, direction: number) {
    const next = [...steps];
    [next[index], next[index + direction]] = [next[index + direction], next[index]];
    onChange(next);
  }
  async function reload(selected: string) {
    const result = await illustrationRequest("bot_mode/tag_filter_profiles") as TagFilterProfiles;
    onSaved(result, selected);
  }
  async function save(name: string) {
    await action.run(async () => {
      await illustrationRequest("bot_mode/tag_filter_profile_save", { name, steps });
      await reload(name);
      setCreating(false); setNewName("");
    }, "필터 프로필을 저장했습니다. 일괄 설정에도 적용됩니다.");
  }
  async function showPreview() {
    await action.run(async () => {
      const targets = await illustrationRequest(`bot_mode/${target === "face" ? "utility_preview" : "rep_preview"}?${query({ bot: botName })}`);
      const items = (targets.images || []).filter((item: IllustrationData) => characters.includes(item.character));
      if (!items.length) throw new Error(target === "face" ? "선택한 캐릭터의 FACE 이미지가 없습니다. 데이터 패치를 먼저 실행하세요." : "선택한 캐릭터의 대표 이미지가 없습니다. 대표 이미지를 먼저 지정하세요.");
      const result = await illustrationRequest("bot_mode/tag_filter_preview", { bot: botName, items, steps });
      setPreview(result.preview || []);
    }, "");
  }

  return <div className="illust-form">
    <div className="illust-preparation-profile-bar">
      <label className="field">{target === "face" ? "얼굴 태그 필터 프로필" : "대표 이미지 태그 필터 프로필"}<select value={profile} disabled={busy || dirty} onChange={event => onSelect(event.target.value)}>
        <option value="">사용 안 함</option>{names.map(name => <option key={name} value={name}>{name} ({filters.profiles[name].length ? `${filters.profiles[name].length}단계` : "규칙 없음"})</option>)}
      </select></label>
      <div className="illust-actions">
        <button type="button" className="button secondary" disabled={busy || invalid} onClick={() => { setCreating(true); setNewName(""); }}><Plus size={14} />새 프로필</button>
        <button type="button" className="button danger" disabled={busy || dirty || !profile || names.length < 2} onClick={() => setDeleting(true)}><Trash2 size={14} />프로필 삭제</button>
      </div>
    </div>
    <p className="field-help">프로필은 여러 봇에서 함께 사용하는 태그 처리 규칙입니다. ‘기본’은 프로필 이름이며, 0단계는 저장된 규칙이 없어 필터링하지 않는다는 뜻입니다.</p>
    <fieldset className="illust-form" disabled={busy}>
      <div className="illust-actions"><h4>필터 규칙 <Badge>{steps.length}단계</Badge></h4>{dirty && <Badge tone="rose">저장하지 않은 변경</Badge>}</div>
      {!steps.length && <p className="field-help">등록된 규칙이 없습니다. 단계 추가로 제거할 패턴이나 바꿀 문자를 설정하세요.</p>}
      {steps.map((step, index) => <section className="illust-filter-rule" key={index} aria-label={`필터 ${index + 1}단계`}>
        <div className="illust-actions"><strong>{index + 1}단계</strong><div className="illust-actions">
          <button type="button" className="button small" aria-label={`${index + 1}단계 위로`} disabled={index === 0} onClick={() => move(index, -1)}><ArrowUp size={14} /></button>
          <button type="button" className="button small" aria-label={`${index + 1}단계 아래로`} disabled={index === steps.length - 1} onClick={() => move(index, 1)}><ArrowDown size={14} /></button>
          <button type="button" className="button danger small" aria-label={`${index + 1}단계 삭제`} onClick={() => onChange(steps.filter((_, i) => i !== index))}><Trash2 size={14} /></button>
        </div></div>
        <div className="illust-fields">
          <label className="field">처리 방식<select value={step.action} onChange={event => onChange(steps.map((current, i) => i === index ? { action: event.target.value, ...(event.target.value === "split_by" ? { separator: "_", take: "0" } : {}) } : current))}>{Object.entries(filterActions).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
          {["remove_match", "strip_prefix", "strip_suffix"].includes(step.action) && <label className="field">정규식 패턴<input value={step.pattern || ""} onChange={event => edit(index, { pattern: event.target.value })} placeholder="제거할 패턴" /></label>}
          {step.action === "replace" && <><label className="field">찾을 문자<input value={step.from || ""} onChange={event => edit(index, { from: event.target.value })} /></label><label className="field">바꿀 문자<input value={step.to || ""} onChange={event => edit(index, { to: event.target.value })} placeholder="비워 두면 삭제" /></label></>}
          {step.action === "split_by" && <><label className="field">구분자<input value={step.separator ?? "_"} onChange={event => edit(index, { separator: event.target.value })} /></label><label className="field">가져올 위치<input value={step.take ?? 0} onChange={event => edit(index, { take: event.target.value })} placeholder="0=앞, -1=끝" /><small>0부터 시작 · -1은 마지막 · 0:2는 앞 두 부분</small></label></>}
        </div>
      </section>)}
      <p className="field-help">구분자 분리는 전체 문자열에 먼저 적용됩니다. 나머지 규칙은 쉼표로 나눈 각 태그에 위에서부터 적용됩니다.</p>
      {invalid && <p className="field-error">각 단계의 패턴, 구분자 또는 찾을 문자를 입력하세요.</p>}
      <div className="illust-actions">
        <button type="button" className="button secondary" onClick={() => onChange([...steps, { action: "remove_match", pattern: "" }])}><Plus size={14} />단계 추가</button>
        <button type="button" className="button" disabled={invalid || !characters.length} onClick={() => void showPreview()}>미리보기</button>
        <button type="button" className="button primary" disabled={!profile || !dirty || invalid} onClick={() => void save(profile)}><Save size={14} />프로필 저장</button>
        {dirty && <button type="button" className="button" onClick={() => onSelect(profile)}>변경 취소</button>}
      </div>
    </fieldset>
    <IllustrationFeedback error={action.error} message={action.message} />
    {preview && <section className="illust-filter-preview" aria-label="태그 필터 미리보기"><h4>미리보기 <Badge>{preview.length}장</Badge></h4>{preview.map((item, index) => <article key={index}><strong>{item.character} · {item.filename}</strong><div className="illust-fields"><div><h5>적용 전</h5><p>{item.original || "태그 없음"}</p></div><div><h5>적용 후</h5><p>{item.filtered || "태그 없음"}</p></div></div></article>)}</section>}
    {creating && <Modal title="새 필터 프로필" backgroundable={false} onClose={() => setCreating(false)} closeDisabled={action.busy}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); if (newName.trim() && !names.includes(newName.trim()) && !invalid) void save(newName.trim()); }}>
        <p className="field-help">현재 편집 중인 규칙 {steps.length}단계를 새 이름으로 저장합니다.</p>
        <label className="field">프로필 이름<input autoFocus value={newName} disabled={action.busy} onChange={event => setNewName(event.target.value)} /></label>
        {names.includes(newName.trim()) && <p className="field-error">같은 이름의 프로필이 있습니다.</p>}
        <IllustrationFeedback error={action.error} />
        <div className="modal-actions"><button type="button" className="button" disabled={action.busy} onClick={() => setCreating(false)}>취소</button><button className="button primary" disabled={action.busy || !newName.trim() || names.includes(newName.trim()) || invalid}>프로필 만들기</button></div>
      </form>
    </Modal>}
    {deleting && <Modal title="필터 프로필 삭제" backgroundable={false} onClose={() => setDeleting(false)} closeDisabled={action.busy}>
      <p>‘{profile}’ 프로필을 삭제할까요?</p><IllustrationFeedback error={action.error} /><div className="modal-actions"><button type="button" className="button" disabled={action.busy} onClick={() => setDeleting(false)}>취소</button><button type="button" className="button danger" disabled={action.busy} onClick={() => void action.run(async () => {
        await illustrationRequest("bot_mode/tag_filter_profile_delete", { name: profile });
        const result = await illustrationRequest("bot_mode/tag_filter_profiles") as TagFilterProfiles;
        onSaved(result, result.active_profile || ""); setDeleting(false);
      }, "필터 프로필을 삭제했습니다.")}>삭제</button></div>
    </Modal>}
  </div>;
}
