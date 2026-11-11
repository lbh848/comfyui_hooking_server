import { useEffect, useId, useMemo, useState, type FormEvent } from "react";
import { CircleHelp, Copy, FilePlus2, Pencil, Plus, Save, Star, Trash2 } from "lucide-react";
import { AssetScopeControls, type AssetScopeAction } from "./AssetScopeControls";
import { Badge, Modal, SettingsTabs } from "./components";
import { emotionGroupKey, emotionGroupRequest, type EmotionPresetGroup, type EmotionPresetGroupCatalog, type EmotionPresetScope } from "./emotionPresetGroups";
import type { IllustrationData } from "./illustration";
import { presetRequest, type PresetCatalog } from "./presets";
import { SearchableSelect } from "./SearchableSelect";

type PresetKind = "sfw" | "nsfw";
type GroupDialogMode = "save" | "duplicate" | "create" | "rename";
export type EmotionPresetRow = { preset: string; name: string };
export const EMOTION_PRESET_CHAR_LIMIT = 10_000;

const presetTabs: Record<PresetKind, string> = { sfw: "SFW", nsfw: "NSFW" };
const presetToggleKeys: Record<PresetKind, string> = {
  sfw: "emotion_presets_sfw",
  nsfw: "emotion_presets_nsfw",
};
const presetDescriptions: Record<PresetKind, string> = {
  sfw: "SFW 장면에 사용할 표정 프리셋과 감정 이름을 연결합니다.",
  nsfw: "NSFW 장면에 사용할 표정 프리셋과 감정 이름을 연결합니다.",
};
const scopeLabels = { builtin: "배포 그룹", local: "사용자 그룹" } as const;
const sameRows = (left: EmotionPresetRow[], right: EmotionPresetRow[]) =>
  left.length === right.length && left.every((row, index) => row.preset === right[index].preset && row.name === right[index].name);

export function EmotionPresetSettings({ toggles, onChange, onTotalCharsChange }: {
  toggles: IllustrationData;
  onChange: (patch: IllustrationData) => void;
  onTotalCharsChange?: (total: number) => void;
}) {
  const representativeHelpId = useId();
  const [kind, setKind] = useState<PresetKind>("sfw");
  const [catalog, setCatalog] = useState<PresetCatalog | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);

  const [groupCatalog, setGroupCatalog] = useState<EmotionPresetGroupCatalog | null>(null);
  const [groupError, setGroupError] = useState("");
  const [groupMessage, setGroupMessage] = useState("");
  const [groupBusy, setGroupBusy] = useState(false);
  const [groupToken, setGroupToken] = useState("");
  const [groupRevision, setGroupRevision] = useState(0);
  const [selection, setSelection] = useState<Record<PresetKind, string>>({ sfw: "", nsfw: "" });
  const [dialog, setDialog] = useState<{ mode: GroupDialogMode; kind: PresetKind } | null>(null);
  const [dialogName, setDialogName] = useState("");
  const [dialogScope, setDialogScope] = useState<EmotionPresetScope>("local");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    presetRequest<PresetCatalog>().then(value => {
      if (active) setCatalog(value);
    }).catch(cause => {
      if (active) setError(cause instanceof Error ? cause.message : "표정 프리셋을 불러오지 못했습니다.");
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [revision]);

  useEffect(() => {
    let active = true;
    setGroupError("");
    emotionGroupRequest<EmotionPresetGroupCatalog>().then(value => {
      if (active) setGroupCatalog(value);
    }).catch(cause => {
      if (active) setGroupError(cause instanceof Error ? cause.message : "감정 프리셋 그룹을 불러오지 못했습니다.");
    });
    return () => { active = false; };
  }, [groupRevision]);

  // The repository catalog is the source for both the dropdown and its tags.
  // A local preset with the same name replaces the distributed entry.
  const presets = useMemo(() => {
    const result = new Map<string, { tags: string[]; scope: "builtin" | "local" }>();
    if (!catalog) return result;
    for (const [scope, entries] of [["builtin", catalog.builtin], ["local", catalog.active]] as const) {
      for (const [name, value] of Object.entries(entries.expressions || {})) {
        if (Array.isArray(value) && value.every(tag => typeof tag === "string")) {
          result.set(name, { tags: value, scope });
        }
      }
    }
    return result;
  }, [catalog]);

  const groups = useMemo(() => (["builtin", "local"] as const).map(scope => ({
    scope,
    label: scope === "builtin" ? "배포 프리셋" : "사용자 프리셋",
    names: [...presets]
      .filter(([, entry]) => entry.scope === scope)
      .map(([name]) => name)
      .sort((a, b) => a.localeCompare(b, "ko", { numeric: true })),
  })).filter(group => group.names.length), [presets]);

  function rows(target: PresetKind): EmotionPresetRow[] {
    const value = toggles[presetToggleKeys[target]];
    return (Array.isArray(value) ? value : [])
      .filter(item => item && typeof item.preset === "string" && typeof item.name === "string")
      .map(item => ({ preset: item.preset, name: item.name }));
  }
  const representative = toggles.emotion_representative && typeof toggles.emotion_representative === "object"
    ? toggles.emotion_representative : null;
  function isRepresentative(target: PresetKind, row: EmotionPresetRow): boolean {
    return !!representative
      && String(representative.rating || "").toLowerCase() === target
      && String(representative.preset || "") === row.preset
      && String(representative.name || "") === row.name;
  }
  function clearsRepresentative(target: PresetKind, next: EmotionPresetRow[]): boolean {
    if (!representative || String(representative.rating || "").toLowerCase() !== target) return false;
    return !next.some(row => row.preset === String(representative.preset || "")
      && row.name === String(representative.name || ""));
  }
  function update(target: PresetKind, next: EmotionPresetRow[]) {
    onChange({ [presetToggleKeys[target]]: next,
      ...(clearsRepresentative(target, next) ? { emotion_representative: {} } : {}) });
  }
  function patch(target: PresetKind, index: number, field: keyof EmotionPresetRow, value: string) {
    const current = rows(target);
    const next = current.map((row, rowIndex) => rowIndex === index ? { ...row, [field]: value } : row);
    onChange({ [presetToggleKeys[target]]: next,
      ...(isRepresentative(target, current[index]) ? { emotion_representative: { ...representative, [field]: value } } : {}) });
  }
  function toggleRepresentative(target: PresetKind, row: EmotionPresetRow) {
    if (!row.name.trim()) return;
    onChange({ emotion_representative: isRepresentative(target, row)
      ? {} : { preset: row.preset, name: row.name, rating: target } });
  }

  // 이 설정은 감정 이름과 표정 태그 목록 그대로 컨텍스트에 들어간다.
  // SFW·NSFW 두 탭의 행을 합친 전체 글자수를 패널 제목 옆에 표시한다.
  const totalChars = useMemo(() => {
    let total = 0;
    for (const target of ["sfw", "nsfw"] as const) {
      for (const row of rows(target)) {
        if (!row.preset && !row.name) continue;
        const tags = presets.get(row.preset)?.tags ?? [];
        total += `${row.name || row.preset}: ${tags.join(", ")}`.length + 1;
      }
    }
    return total;
  }, [presets, toggles]);
  useEffect(() => { onTotalCharsChange?.(totalChars); }, [onTotalCharsChange, totalChars]);

  function groupList(target: PresetKind, scope: EmotionPresetScope): EmotionPresetGroup[] {
    return groupCatalog?.[scope]?.[target] ?? [];
  }
  function selectedGroup(target: PresetKind): (EmotionPresetGroup & { scope: EmotionPresetScope }) | null {
    const key = selection[target];
    if (!key) return null;
    for (const scope of ["builtin", "local"] as const) {
      const group = groupList(target, scope).find(item => emotionGroupKey(scope, item.name) === key);
      if (group) return { ...group, scope };
    }
    return null;
  }

  // 첫 로드 시 현재 표와 똑같은 저장 그룹이 있으면 자동으로 선택해 둔다.
  useEffect(() => {
    if (!groupCatalog) return;
    setSelection(current => {
      let next = current;
      for (const target of ["sfw", "nsfw"] as const) {
        if (next[target]) continue;
        const items = rows(target);
        const match = (["builtin", "local"] as const).flatMap(scope =>
          groupList(target, scope).map(group => ({ group, scope })))
          .find(({ group }) => group.rows.length === items.length
            && group.rows.every((row, index) => row.preset === items[index].preset && row.name === items[index].name));
        if (match) next = { ...next, [target]: emotionGroupKey(match.scope, match.group.name) };
      }
      return next;
    });
    // 현재 표가 바뀔 때마다 다시 선택하지 않도록 그룹 목록 변경에만 반응한다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groupCatalog]);

  async function runGroup(action: () => Promise<boolean>): Promise<boolean> {
    setGroupBusy(true);
    setGroupError("");
    try { return await action(); }
    catch (cause) {
      setGroupError(cause instanceof Error ? cause.message : "감정 프리셋 그룹을 변경할 수 없습니다.");
      return false;
    } finally { setGroupBusy(false); }
  }

  function scopeAction(action: AssetScopeAction): Promise<boolean> {
    if (action === "scope") return Promise.resolve(false);
    return runGroup(async () => {
      const result = await emotionGroupRequest<EmotionPresetGroupCatalog>(action, {
        confirmed: action === "unlock", builtin_unlock_token: groupToken,
      });
      setGroupCatalog(result);
      setGroupToken(action === "unlock" ? result.builtin_unlock_token || "" : "");
      setGroupMessage(action === "unlock" ? "배포 그룹을 편집할 수 있습니다." : "배포 그룹을 잠갔습니다.");
      return true;
    });
  }

  function selectGroup(target: PresetKind, keyValue: string) {
    setGroupError("");
    setGroupMessage("");
    if (!keyValue) { setSelection(current => ({ ...current, [target]: "" })); return; }
    let found: (EmotionPresetGroup & { scope: EmotionPresetScope }) | null = null;
    for (const scope of ["builtin", "local"] as const) {
      const group = groupList(target, scope).find(item => emotionGroupKey(scope, item.name) === keyValue);
      if (group) { found = { ...group, scope }; break; }
    }
    if (!found) return;
    const current = rows(target);
    const same = current.length === found.rows.length
      && found.rows.every((row, index) => row.preset === current[index].preset && row.name === current[index].name);
    if (!same && current.length && !window.confirm("현재 표를 선택한 프리셋 그룹의 내용으로 바꿀까요?")) return;
    update(target, found.rows.map(row => ({ ...row })));
    setSelection(state => ({ ...state, [target]: keyValue }));
  }

  function copyName(target: PresetKind, group: EmotionPresetGroup): string {
    const taken = new Set(groupList(target, "local").map(item => item.name));
    const base = `${group.name} 복사`;
    let candidate = base;
    let index = 2;
    while (taken.has(candidate)) candidate = `${base} ${index++}`;
    return candidate;
  }

  function openDialog(target: PresetKind, mode: GroupDialogMode) {
    const group = selectedGroup(target);
    if ((mode === "duplicate" || mode === "rename") && !group) return;
    setGroupError("");
    setGroupMessage("");
    setDialogScope(mode === "save" ? group?.scope ?? "local" : "local");
    setDialogName(mode === "create" ? "" : mode === "duplicate" && group ? copyName(target, group) : group?.name ?? "");
    setDialog({ mode, kind: target });
  }

  function deleteGroup(target: PresetKind) {
    const group = selectedGroup(target);
    if (!group || (group.scope === "builtin" && !groupToken)) return;
    if (!window.confirm(`'${group.name}' 프리셋 그룹을 삭제할까요?`)) return;
    void runGroup(async () => {
      const result = await emotionGroupRequest<EmotionPresetGroupCatalog>("delete", {
        kind: target, scope: group.scope, name: group.name, confirmed: true,
        builtin_unlock_token: groupToken,
      });
      setGroupCatalog(result);
      setSelection(state => ({ ...state, [target]: "" }));
      setGroupMessage("감정 프리셋 그룹을 삭제했습니다.");
      return true;
    });
  }

  async function submitDialog(event: FormEvent) {
    event.preventDefault();
    if (!dialog || !dialogName.trim()) return;
    const { mode, kind: target } = dialog;
    const name = dialogName.trim();
    const selected = selectedGroup(target);
    const scope = mode === "duplicate" ? "local" : mode === "rename" ? selected?.scope ?? "local" : dialogScope;
    await runGroup(async () => {
      let result: EmotionPresetGroupCatalog;
      if (mode === "create") {
        result = await emotionGroupRequest("create", {
          kind: target, scope, name, builtin_unlock_token: groupToken,
        });
        update(target, []);
        setGroupMessage("새 감정 프리셋 그룹을 만들었습니다.");
      } else if (mode === "duplicate") {
        if (!selected) throw new Error("복제할 프리셋 그룹을 먼저 선택하세요.");
        result = await emotionGroupRequest("duplicate", {
          kind: target, source_scope: selected.scope, source_name: selected.name,
          new_name: name, builtin_unlock_token: groupToken,
        });
        setGroupMessage("선택한 그룹을 사용자 그룹에 복제했습니다.");
      } else if (mode === "rename") {
        if (!selected) throw new Error("이름을 바꿀 프리셋 그룹을 먼저 선택하세요.");
        result = await emotionGroupRequest("rename", {
          kind: target, scope: selected.scope, name: selected.name,
          new_name: name, builtin_unlock_token: groupToken,
        });
        setGroupMessage("감정 프리셋 그룹 이름을 바꿨습니다.");
      } else {
        if (groupList(target, scope).some(group => group.name === name)
          && !window.confirm(`'${name}' 프리셋 그룹을 덮어쓸까요?`)) return false;
        result = await emotionGroupRequest("save", {
          kind: target, scope, name, rows: rows(target), builtin_unlock_token: groupToken,
        });
        setGroupMessage("감정 프리셋 그룹을 저장했습니다.");
      }
      setGroupCatalog(result);
      setSelection(state => ({ ...state, [target]: emotionGroupKey(scope, name) }));
      setDialog(null);
      return true;
    });
  }

  const dialogTitle = dialog?.mode === "duplicate" ? "감정 프리셋 복제"
    : dialog?.mode === "create" ? "새 감정 프리셋 그룹"
    : dialog?.mode === "rename" ? "감정 프리셋 이름 바꾸기" : "감정 프리셋 저장";
  const dialogSubmit = dialog?.mode === "duplicate" ? "복제"
    : dialog?.mode === "create" ? "새로 만들기"
    : dialog?.mode === "rename" ? "이름 바꾸기" : "저장";

  return <div className="illust-body illust-generation-layout">
    <SettingsTabs id="emotion-presets" orientation="vertical" className="illust-generation-tabs"
      label="프리셋 설정 구분" values={presetTabs} value={kind} onChange={value => setKind(value as PresetKind)} />
    <div className="illust-generation-main">
      {error && <p className="field-error" role="alert">{error}<button type="button" className="button" disabled={loading} onClick={() => setRevision(current => current + 1)}>다시 시도</button></p>}
      {loading && !catalog && <p className="field-help" role="status">저장소 표정 프리셋을 불러오는 중…</p>}
      {(["sfw", "nsfw"] as const).map(target => {
        const items = rows(target);
        const selected = selectedGroup(target);
        const locked = selected?.scope === "builtin" && !groupToken;
        const dirty = !!selected && !sameRows(items, selected.rows);
        return <div key={target} role="tabpanel" id={`emotion-presets-panel-${target}`}
          aria-labelledby={`emotion-presets-tab-${target}`} hidden={kind !== target} className="illust-generation-content">
          <header className="illust-generation-heading"><h3>{presetTabs[target]}</h3><p>{presetDescriptions[target]}</p></header>
          <div className="illust-emotion-preset-groups">
            <div className="illust-emotion-preset-group-row">
              <label className="field illust-emotion-preset-group-select">프리셋 그룹
                <SearchableSelect aria-label={`${presetTabs[target]} 감정 프리셋 그룹`} value={selection[target]}
                  disabled={groupBusy} onChange={event => selectGroup(target, event.target.value)}>
                  {(["builtin", "local"] as const).map(scope => <optgroup key={scope} label={scopeLabels[scope]}>
                    {groupList(target, scope).length
                      ? groupList(target, scope).map(group => <option key={group.name} value={emotionGroupKey(scope, group.name)}>{group.name} ({group.rows.length})</option>)
                      : <option disabled value="">없음</option>}
                  </optgroup>)}
                </SearchableSelect>
              </label>
              <Badge>{selected ? `${selected.scope === "builtin" ? (groupToken ? "배포 · 편집 가능" : "배포 · 읽기 전용") : "사용자"} · ${selected.rows.length}행${dirty ? " · 수정됨" : ""}` : "저장 안 됨"}</Badge>
              <AssetScopeControls unlocked={!!groupToken} disabled={groupBusy} onAction={scopeAction} />
            </div>
            <div className="illust-emotion-preset-actions">
              <div className="asset-actions">
                <button type="button" className="button primary" disabled={groupBusy || locked} onClick={() => openDialog(target, "save")}><Save size={14} />저장</button>
                <button type="button" className="button" disabled={groupBusy || !selected} onClick={() => openDialog(target, "duplicate")}><Copy size={14} />복제</button>
                <button type="button" className="button" disabled={groupBusy || !selected || locked} onClick={() => openDialog(target, "rename")}><Pencil size={14} />이름 바꾸기</button>
                <button type="button" className="button" disabled={groupBusy} onClick={() => openDialog(target, "create")}><FilePlus2 size={14} />새로 만들기</button>
              </div>
              <button type="button" className="button danger" disabled={groupBusy || !selected || locked} onClick={() => deleteGroup(target)}><Trash2 size={14} />삭제</button>
            </div>
          </div>
          {groupError && <p className="settings-feedback error" role="alert">{groupError}
            <button type="button" className="button" disabled={groupBusy} onClick={() => setGroupRevision(current => current + 1)}>다시 시도</button>
          </p>}
          {groupMessage && <p className="settings-feedback" role="status">{groupMessage}</p>}
          <div className="illust-emotion-preset-table-wrap">
            <table className="illust-emotion-preset-table">
              <thead><tr><th scope="col">표정 감정</th><th scope="col">감정 이름</th>
                <th scope="col" className="illust-emotion-preset-help-head">
                  <span className="resource-trigger-help illust-emotion-preset-help">
                    <button type="button" className="icon-button" aria-label="복장 대표 표정 도움말"
                      aria-describedby={`${representativeHelpId}-help`}>
                      <CircleHelp size={15} aria-hidden="true" />
                    </button>
                    <span id={`${representativeHelpId}-help`} className="resource-trigger-tooltip" role="tooltip">
                      지정한 감정을 복장 대표를 만들기 위한 감정으로 사용합니다.
                    </span>
                  </span>
                </th>
                <th scope="col"><span className="sr-only">삭제</span></th></tr></thead>
              <tbody>
                {items.map((row, index) => {
                  const entry = row.preset ? presets.get(row.preset) : undefined;
                  return <tr key={index}>
                    <td>
                      <SearchableSelect aria-label={`${presetTabs[target]} 표정 감정 ${index + 1}`} value={row.preset}
                        onChange={event => patch(target, index, "preset", event.target.value)}>
                        <option value="">표정 감정 선택</option>
                        {groups.map(group => <optgroup key={group.scope} label={group.label}>
                          {group.names.map(name => <option key={name} value={name}>{name}</option>)}
                        </optgroup>)}
                        {row.preset && !presets.has(row.preset) && <option value={row.preset}>{row.preset} (저장소에 없음)</option>}
                      </SearchableSelect>
                      {row.preset && <div className="illust-emotion-preset-tagbox">
                        {!entry ? <span className="field-help">저장소에서 이 프리셋을 찾을 수 없습니다.</span>
                          : entry.tags.length ? <div className="illust-tag-list">{entry.tags.map((tag, tagIndex) => <span key={`${tag}-${tagIndex}`}>{tag}</span>)}</div>
                          : <span className="field-help">이 프리셋에는 등록된 태그가 없습니다.</span>}
                      </div>}
                    </td>
                    <td>
                      <input className="illust-emotion-preset-name" aria-label={`${presetTabs[target]} 감정 이름 ${index + 1}`}
                        value={row.name} placeholder={row.preset || "감정 이름 입력"}
                        onChange={event => patch(target, index, "name", event.target.value)} />
                    </td>
                    <td>
                      <button type="button" className="icon-button" aria-label={`${row.name || "감정"} 복장 대표 표정으로 지정`}
                        aria-pressed={isRepresentative(target, row)} title="복장 대표 표정으로 지정"
                        disabled={!row.name.trim()} onClick={() => toggleRepresentative(target, row)}>
                        <Star size={14} fill={isRepresentative(target, row) ? "currentColor" : "none"} /></button>
                    </td>
                    <td>
                      <button type="button" className="icon-button" aria-label={`${index + 1}번째 행 삭제`}
                        onClick={() => update(target, items.filter((_, rowIndex) => rowIndex !== index))}><Trash2 size={14} /></button>
                    </td>
                  </tr>;
                })}
              </tbody>
            </table>
            {!items.length && <p className="field-help">아직 추가한 표정 프리셋이 없습니다. 아래 + 버튼으로 행을 추가하세요.</p>}
            <p className="field-help">별표한 감정은 새 복장·새 NPC의 대표 이미지를 먼저 생성할 때 사용됩니다. 새 복장 커버는 캐릭터 자동 생성 및 복장 이미지 생성 모드, 새 NPC는 자동 생성 설정을 따릅니다. 이미 이미지가 있는 복장에는 다시 생성하지 않습니다.</p>
            <button type="button" className="button secondary illust-emotion-preset-add"
              onClick={() => update(target, [...items, { preset: "", name: "" }])}><Plus size={14} />행 추가</button>
          </div>
        </div>;
      })}
    </div>
    {dialog && <Modal title={dialogTitle} backgroundable={false} closeDisabled={groupBusy} onClose={() => setDialog(null)}>
      <form className="asset-form" onSubmit={event => void submitDialog(event)}>
        <label className="field">프리셋 그룹 이름
          <input required autoFocus value={dialogName} disabled={groupBusy}
            onChange={event => setDialogName(event.target.value)} />
        </label>
        {(dialog.mode === "create" || dialog.mode === "save") && groupToken &&
          <label className="field">그룹
            <select value={dialogScope} disabled={groupBusy}
              onChange={event => setDialogScope(event.target.value as EmotionPresetScope)}>
              <option value="builtin">배포 그룹</option><option value="local">사용자 그룹</option>
            </select>
          </label>}
        {dialog.mode === "create" && <p className="field-help">새 그룹을 만들면 현재 표가 비워집니다. 행을 채운 뒤 저장하세요.</p>}
        {dialog.mode === "duplicate" && <p className="field-help">선택한 그룹을 사용자 그룹에 복제합니다.</p>}
        {dialog.mode === "rename" && <p className="field-help">프리셋 그룹의 이름만 바꿉니다. 행 구성은 그대로 유지됩니다.</p>}
        {dialog.mode === "save" && <p className="field-help">현재 {presetTabs[dialog.kind]} 표의 {rows(dialog.kind).length}개 행을 저장합니다.</p>}
        {groupError && <p className="settings-feedback error" role="alert">{groupError}</p>}
        <div className="asset-actions">
          <button type="button" className="button" disabled={groupBusy} onClick={() => setDialog(null)}>취소</button>
          <button type="submit" className="button primary" disabled={groupBusy || !dialogName.trim()}>
            {groupBusy ? "처리 중…" : dialogSubmit}
          </button>
        </div>
      </form>
    </Modal>}
  </div>;
}
