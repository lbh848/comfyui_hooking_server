import { referenceStageDelivery, referenceStageDeliveryOptions, referencePreprocessOptions } from "./referenceSettings";
import { useEffect, useId, useRef, useState } from "react";
import { Copy, Dice5, RotateCcw } from "lucide-react";
import { Badge, Modal, SettingsTabs } from "./components";
import { TagTextarea } from "./TagTextarea";
import { SearchableSelect } from "./SearchableSelect";
import { IllustrationLoading } from "./IllustBackups";
import { query, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import { presetRequest, valueText, type PresetCatalog } from "./presets";
import { useWorkflowAutosave } from "./useWorkflowAutosave";
import { IllustGenerationPresets } from "./IllustGenerationPresets";

const allGenerationModes = { illustration: "일반 삽화 생성 모드", character_auto: "캐릭터 자동 생성 및 복장 이미지 생성 모드", background: "배경 생성 모드" };
const illustrationModes = { illustration: "일반 삽화 생성 모드", character_auto: "캐릭터 자동 생성 및 복장 이미지 생성 모드" };
const emotionModes = { illustration: "에셋 생성 모드", character_auto: "캐릭터 자동 생성 및 복장 이미지 생성 모드", background: "배경 생성 모드" };
type GenerationMode = keyof typeof allGenerationModes;
const profiles = { solo: "1인", group: "2인 이상" };
type Profile = keyof typeof profiles;
type Settings = Record<GenerationMode, Record<Profile, IllustrationData>>;
const families = { anima: "ANIMA", sdxl: "ILXL" };
const presetKinds = { artist: "아티스트", quality: "품질", background: "배경", negative: "부정" };
const presetFields = Object.entries(families).flatMap(([family, label]) => Object.entries(presetKinds).map(([kind, title]) => ({
  key: `${family}_${kind}_preset`, label: `${label} ${title}`, category: `${kind}_presets`, family,
})));
const changes = (value: IllustrationData, base: IllustrationData) => Object.fromEntries(Object.entries(value).filter(([key, item]) => item !== base[key]));
const imageSizeSeed = ({ img_w, img_h, seed }: IllustrationData) => ({ img_w, img_h, seed });
const settingFields = ({ generation_mode: _mode, ...settings }: IllustrationData) => settings;
// 배율은 x2, x0.3처럼 표시하고, 계산된 픽셀 크기는 항상 정수·16배수로 맞춘다.
const scaleValue = (value: unknown) => {
  const number = Number(value ?? 1);
  return Number.isFinite(number) ? Math.min(2, Math.max(0.2, number)) : 1;
};
const scaleLabel = (value: number) => `x${Number.isInteger(value) ? value : value.toFixed(1)}`;

export function IllustBotGenerationSettings({ botName, readOnly, onDirtyChange, onBusyChange, initialMode = "illustration" }: {
  botName: string; readOnly: boolean; onDirtyChange: (dirty: boolean) => void; onBusyChange: (busy: boolean) => void;
  initialMode?: GenerationMode;
}) {
  const api = useBotApi();
  const generationModes = (api.scope === "emotion" ? emotionModes : illustrationModes) as Record<GenerationMode, string>;
  const modeKeys = Object.keys(generationModes) as GenerationMode[];
  /** 에셋 생성 모드와 배경 생성 모드는 1인 전용이다. */
  const groupLocked = (generationMode: GenerationMode) =>
    generationMode === "background" || (api.scope === "emotion" && generationMode === "illustration");
  const id = useId();
  const [mode, setMode] = useState<GenerationMode>(initialMode);
  const solo = useBotResource(`bot_mode/illust_settings?${query({ bot_name: botName, profile: "solo" })}`);
  const group = useBotResource(`bot_mode/illust_settings?${query({ bot_name: botName, profile: "group" })}`);
  const autoSolo = useBotResource(`bot_mode/illust_settings?${query({ bot_name: botName, generation_mode: "character_auto", profile: "solo" })}`);
  const backgroundSolo = useBotResource(`bot_mode/illust_settings?${query({ bot_name: botName, generation_mode: "background", profile: "solo" })}`);
  const backgroundGroup = useBotResource(`bot_mode/illust_settings?${query({ bot_name: botName, generation_mode: "background", profile: "group" })}`);
  const config = useBotResource("illustration/config");
  const toggles = useBotResource("illustration_context/toggles");
  const tags = useBotResource("asset_mode/tags", true);
  const [draft, setDraft] = useState<Settings | null>(null);
  const [base, setBase] = useState<Settings | null>(null);
  const [profile, setProfile] = useState<Profile>("solo");
  const [copy, setCopy] = useState(false);
  const [editor, setEditor] = useState<{ mode: GenerationMode; field: string } | null>(null);
  const [editorBusy, setEditorBusy] = useState(false);
  const [presetBusy, setPresetBusy] = useState(false);
  const form = useRef<HTMLFormElement>(null);
  useEffect(() => {
    if (solo.data && group.data && autoSolo.data?.generation_mode === "character_auto"
        && backgroundSolo.data?.generation_mode === "background" && backgroundGroup.data) {
      const loaded = {
        illustration: { solo: settingFields(solo.data), group: imageSizeSeed(group.data) },
        character_auto: { solo: settingFields(autoSolo.data), group: imageSizeSeed(autoSolo.data) },
        background: { solo: settingFields(backgroundSolo.data), group: imageSizeSeed(backgroundGroup.data) },
      };
      setDraft(loaded); setBase(loaded);
    }
  }, [solo.data, group.data, autoSolo.data, backgroundSolo.data, backgroundGroup.data]);
  const dirty = !!draft && !!base && JSON.stringify(draft) !== JSON.stringify(base);
  const autosave = useWorkflowAutosave({ label: `${botName} · 그리기 설정`, value: JSON.stringify(draft), dirty: dirty && !readOnly, form, save: async () => {
    if (!draft || !base || readOnly) return;
    for (const generationMode of modeKeys) {
      for (const key of Object.keys(profiles) as Profile[]) {
        if (generationMode === "character_auto" && key === "group") continue;
        const patch = changes(draft[generationMode][key], base[generationMode][key]);
        if (!Object.keys(patch).length) continue;
        await api.request("bot_mode/update_illust_settings", { bot_name: botName, generation_mode: generationMode, profile: key, illust_settings: patch });
        setBase(current => current && ({ ...current, [generationMode]: { ...current[generationMode], [key]: draft[generationMode][key] } }));
      }
    }
  } });
  useEffect(() => { if (groupLocked(mode)) setProfile("solo"); }, [mode, api.scope]);
  useEffect(() => { onDirtyChange(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => { onBusyChange(autosave.busy || editorBusy || presetBusy); }, [autosave.busy, editorBusy, presetBusy, onBusyChange]);
  useEffect(() => () => { onDirtyChange(false); onBusyChange(false); }, [onDirtyChange, onBusyChange]);
  const resources = [solo, group, autoSolo, backgroundSolo, backgroundGroup, config, toggles, tags];
  const unsupported = (!!autoSolo.data && autoSolo.data.generation_mode !== "character_auto")
    || (!!backgroundSolo.data && backgroundSolo.data.generation_mode !== "background");
  if (!draft || !config.data || !toggles.data || !tags.data || unsupported) return <IllustrationLoading error={unsupported ? "모드별 그리기 설정을 사용하려면 프로그램을 다시 시작하세요." : resources.find(item => item.error)?.error || ""} retry={() => resources.forEach(item => item.reload())} />;
  const capabilities = config.data.capabilities || {};
  const mangaPageSizes = config.data.manga_page_sizes;
  const presetTags = tags.data;
  const manga = ["manga", "manga_both"].includes(toggles.data.toggles.illustration_output_mode);
  const target = profile === "solo" ? "group" : "solo";
  const update = (generationMode: GenerationMode, key: Profile, field: string, value: unknown) => { if (!readOnly) setDraft(current => current && ({
    ...current, [generationMode]: { ...current[generationMode], [key]: { ...current[generationMode][key], [field]: value } },
  })); };
  return <div className="illust-bot-generation illust-form">
    <fieldset disabled={presetBusy} className="illust-form"><SettingsTabs id={id} className="illust-bot-generation-mode-tabs" label="그리기 설정 모드" values={generationModes} value={mode} onChange={value => setMode(value as GenerationMode)} /></fieldset>
    <div className="illust-bot-generation-toolbar">
      <span className="field-help">현재 그림 모델: {capabilities.moss ? "MOSS" : config.data.workflow_types[config.data.config.illustration_workflow_type]}</span>
      <span className="field-help" role="status">{autosave.status}</span>
    </div>
    {(autosave.error || autosave.inputError) && <div className="notice error" role="alert">{autosave.error || autosave.inputError}{autosave.error && <button type="button" className="button" onClick={() => void autosave.retry()}>저장 다시 시도</button>}</div>}
    <IllustGenerationPresets key={`${botName}/${mode}`} mode={mode} label={generationModes[mode]} value={draft[mode]} readOnly={readOnly}
      busy={autosave.busy || editorBusy} invalid={!!autosave.inputError} onBusyChange={setPresetBusy}
      onApply={settings => {
        if (readOnly) return;
        setDraft(current => current && ({ ...current, [mode]: {
          solo: { ...current[mode].solo, snapref_primary_reference: "", snapref_detailer_reference: "", ...settings.solo },
          group: mode === "character_auto" || groupLocked(mode) ? current[mode].group : { ...current[mode].group, ...settings.group },
        } }));
      }} />
    <form ref={form} className="illust-form" onSubmit={event => event.preventDefault()}>
      {modeKeys.map(generationMode => {
        const mangaMode = generationMode === "illustration" && manga;
        return <div key={generationMode} id={`${id}-panel-${generationMode}`} role="tabpanel" aria-labelledby={`${id}-tab-${generationMode}`} hidden={mode !== generationMode} className="illust-form">
        <CommonGenerationFields value={draft[generationMode].solo} tags={presetTags} capabilities={capabilities} readOnly={readOnly}
          update={(field, value) => update(generationMode, "solo", field, value)} onEdit={field => setEditor({ mode: generationMode, field })} />
        <section className="illust-bot-generation-section illust-form" data-group="image-size-seed"><h3>이미지 크기·시드</h3>
          {generationMode !== "character_auto" && !mangaMode && <div className="illust-bot-generation-toolbar">
            <SettingsTabs id={`${id}-${generationMode}-size`} label="이미지 크기·시드 인원 선택" values={profiles} value={profile} onChange={value => setProfile(value as Profile)} disabled={groupLocked(generationMode) ? ["group"] : []} />
            {!groupLocked(generationMode) && <button type="button" className="button" disabled={readOnly || autosave.busy || !!autosave.inputError} onClick={() => setCopy(true)}><Copy size={14} />{profiles[target]}으로 복사</button>}
          </div>}
          {generationMode === "character_auto" ? <ImageSizeSeedFields value={draft[generationMode].solo} capabilities={capabilities} manga={false} readOnly={readOnly} update={(field, value) => update(generationMode, "solo", field, value)} />
            : mangaMode ? <ImageSizeSeedFields value={draft[generationMode].solo} capabilities={capabilities} manga readOnly={readOnly} update={(field, value) => update(generationMode, "solo", field, value)} />
            : (Object.keys(profiles) as Profile[]).map(key => <div key={key} id={`${id}-${generationMode}-size-panel-${key}`} role="tabpanel" aria-labelledby={`${id}-${generationMode}-size-tab-${key}`} hidden={profile !== key}>
              <ImageSizeSeedFields value={draft[generationMode][key]} capabilities={capabilities} manga={false} readOnly={readOnly} update={(field, value) => update(generationMode, key, field, value)} />
            </div>)}
        </section>
        {generationMode === "illustration" && <MangaSizeSeedFields value={draft[generationMode].solo} pageSizes={mangaPageSizes} readOnly={readOnly} locked={!mangaMode} update={(field, value) => update(generationMode, "solo", field, value)} />}
        </div>;
      })}
    </form>
    {copy && <Modal title={`${profiles[target]} 이미지 크기·시드로 복사`} closeDisabled={autosave.busy} backgroundable={false} onClose={() => setCopy(false)}>
      <div className="illust-form"><p>{generationModes[mode]}의 {profiles[profile]} 너비·높이·시드를 같은 모드의 {profiles[target]} 설정에 복사합니다.</p>
        <div className="illust-actions"><button className="button" onClick={() => setCopy(false)}>취소</button><button className="button primary" disabled={readOnly || autosave.busy} onClick={() => {
          setDraft(current => current && ({ ...current, [mode]: { ...current[mode], [target]: { ...current[mode][target], ...imageSizeSeed(current[mode][profile]) } } })); setCopy(false); setProfile(target);
        }}>복사</button></div>
      </div>
    </Modal>}
    {editor && <GenerationPresetEditor field={presetFields.find(field => field.key === editor.field)!} selected={String(draft[editor.mode].solo[editor.field] || "")} readOnly={readOnly}
      onClose={() => setEditor(null)} onBusyChange={setEditorBusy} onSaved={name => { update(editor.mode, "solo", editor.field, name); tags.reload(); setEditor(null); }} />}
  </div>;
}

function CommonGenerationFields({ value, tags, capabilities, readOnly, update, onEdit }: {
  value: IllustrationData; tags: IllustrationData; capabilities: IllustrationData; readOnly: boolean;
  update: (field: string, value: unknown) => void; onEdit: (field: string) => void;
}) {
  const check = (key: string, label: string, checked = !!value[key]) => <label className="illust-check"><input type="checkbox" checked={checked} onChange={event => update(key, event.target.checked)} />{label}</label>;
  const range = (key: string, label: string, min: number, max: number, step: number, unit = "") => <label className="field illust-bot-generation-range"><span>{label}<output>{value[key]}{unit}</output></span><input type="range" aria-label={label} min={min} max={max} step={step} value={value[key]} onChange={event => update(key, Number(event.target.value))} /></label>;
  return <fieldset className="illust-form" disabled={readOnly}>
    <section className="illust-bot-generation-section" data-group="presets"><h3>프롬프트 프리셋</h3><div className="illust-bot-generation-columns">
      {Object.entries(families).map(([family, label]) => <fieldset className="illust-form illust-bot-generation-family" data-family={family} disabled={!capabilities[`${family}_presets`]} key={family}>
        <legend>{label}{!capabilities[`${family}_presets`] && <span>현재 그림 모델에서 미사용</span>}</legend>
        {presetFields.filter(field => field.family === family).map(field => {
          const choices = tags[field.category] || {};
          const selected = value[field.key] || "";
          const selectedTags = choices[selected];
          return <div className="illust-bot-generation-preset" key={field.key}>
            <label className="field">{field.label} 프리셋<SearchableSelect aria-label={`${field.label} 프리셋`} value={selected} onChange={event => update(field.key, event.target.value)}>
              <option value="">없음</option>{Object.keys(choices).sort((a, b) => a.localeCompare(b, "ko", { numeric: true })).map(name => <option key={name}>{name}</option>)}
              {selected && !(selected in choices) && <option value={selected}>{selected} · 찾을 수 없음</option>}
            </SearchableSelect></label>
            <div className="illust-bot-generation-tags" role="group" aria-label={`${field.label} 태그`} tabIndex={0}>{Array.isArray(selectedTags) && selectedTags.length ? selectedTags.map((tag: string, index: number) => <span key={index}>{tag}</span>) : <p className="field-help">{selected ? selectedTags ? "태그 없음" : "프리셋을 찾을 수 없습니다." : "선택한 프리셋 없음"}</p>}</div>
            <button type="button" className="button" aria-label={`${field.label} 태그 편집`} onClick={() => onEdit(field.key)}>태그 편집</button>
          </div>;
        })}
      </fieldset>)}
    </div></section>
    <fieldset className="illust-bot-generation-section illust-form" data-group="reference" disabled={capabilities.moss || (!capabilities.local_advanced || !capabilities.anima_presets)}><legend>이미지 레퍼런스{capabilities.moss && <span>CharRef v2 · 사용 중지</span>}</legend>
      {check("snapref_enabled", "이미지 레퍼런스 사용", !capabilities.moss && !!value.snapref_enabled)}
      {!capabilities.moss && value.snapref_enabled && <div className="illust-form illust-bot-generation-dependent">
        {!capabilities.moss_reference && <label className="field">레퍼런스 전처리<select value={value.snapref_preprocess || "none"} onChange={event => update("snapref_preprocess", event.target.value)}>{referencePreprocessOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>}
        {range("snapref_strength", "Strength · 레퍼런스 강도", 0, 2, .05)}
        {!capabilities.moss_reference && range("snapref_fidelity", "Fidelity · 디테일 유지", 0, 1, .05)}
        <div className="illust-bot-generation-columns">
          {([["primary", "1st sampler/디테일러/HD"], ["detailer", "FD/ED"]] as const).map(([stage, label]) =>
            <label key={stage} className="field">레퍼런스 전달 · {label}<select value={referenceStageDelivery(value, stage)} onChange={event => update(`snapref_${stage}_reference`, event.target.value)}>
              {referenceStageDeliveryOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select></label>)}
        </div>
        {!capabilities.moss_reference && <div className="illust-bot-generation-columns"><label className="field">참조 방식<select value={value.snapref_mode} onChange={event => update("snapref_mode", event.target.value)}><option value="char">캐릭터 외형</option><option value="cs">외형 + 화풍</option></select></label>
          {range("snapref_megapixels", "레퍼런스 해상도", .25, 2, .25, " MP")}</div>}
      </div>}
    </fieldset>
    <fieldset className="illust-bot-generation-section illust-form" data-group="enhancement" disabled={capabilities.moss || !capabilities.local_advanced}><legend>고해상도·디테일러 보정{capabilities.moss && <span>MOSS 미사용</span>}</legend>
      {range("hrf_size", "HIRESFIX 및 디테일러 적용 전 확대 배율", 1, 3, .05, "배")}
      {check("hrf_restore_size", "원본 크기 복원")}
      <div className="illust-bot-generation-columns">{Object.entries(families).map(([family, label]) => <fieldset key={family} className="illust-form illust-bot-generation-family" data-family={family} disabled={!capabilities[`${family}_presets`]}><legend>{label}</legend>
        {check(family === "anima" ? "anima_hrf_activate" : "hrf_activate", `${label} HIRESFIX`)}
        <div className="illust-actions">{[["fd", "얼굴"], ["hd", "손"], ["ed", "눈"]].map(([kind, title]) => <span key={kind}>{check(`${family === "anima" ? "anima_" : ""}${kind}_activate`, `${title} 보정 (${kind.toUpperCase()})`)}</span>)}</div>
      </fieldset>)}</div>
      <GenerationNumberField label="FD / ED 시 고정 사이즈 (px)" value={value.face_lora_upscale_size} min={0} optional onChange={next => update("face_lora_upscale_size", next)} />
    </fieldset>
    {capabilities.moss_reference && <p className="field-help">MOSS의 이미지 레퍼런스(CharRef v2)는 현재 사용을 중지했습니다.</p>}
    {capabilities.moss && !capabilities.moss_reference && <p className="field-help">MOSS 그림 모델에서는 이미지 레퍼런스·고해상도 보정·얼굴·손·눈 디테일러를 사용하지 않습니다.</p>}
    {!capabilities.local_advanced && !capabilities.moss_reference && <p className="field-help">현재 그림 모델에서는 로컬 전용 시드·레퍼런스·고해상도·디테일러 설정을 사용하지 않습니다.</p>}
  </fieldset>;
}

function GenerationNumberField({ label, value, min, max, optional = false, snapToStep, onChange }: {
  label: string; value: number | string; min: number; max?: number; optional?: boolean; snapToStep?: number; onChange: (value: number | string) => void;
}) {
  const [focused, setFocused] = useState(false);
  const step = snapToStep ?? 1;
  // Untouched saved sizes may predate the step rule; keep them valid until edited.
  const inputStep = snapToStep && !focused && Number(value) % step !== 0 ? "any" : step;
  return <label className="field">{label}<input type="number" min={min} max={max} step={inputStep} required={!optional} value={value ?? ""} placeholder={optional ? "미사용" : undefined}
    onFocus={() => setFocused(true)}
    onChange={event => onChange(event.target.value === "" ? "" : Number(event.target.value))}
    onBlur={event => {
      setFocused(false);
      const number = event.currentTarget.valueAsNumber;
      if (snapToStep && Number.isFinite(number)) onChange(Math.min(max ?? Infinity, Math.max(min, Math.round(number / step) * step)));
    }}
    onKeyDown={event => { if (snapToStep && event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); } }}
  /></label>;
}

function ImageSizeSeedFields({ value, capabilities, manga, readOnly, update }: {
  value: IllustrationData; capabilities: IllustrationData; manga: boolean; readOnly: boolean; update: (field: string, value: unknown) => void;
}) {
  return <fieldset className={`illust-form${manga ? " illust-locked" : ""}`} disabled={readOnly || manga} aria-disabled={readOnly || manga || undefined}>
    <fieldset className="illust-bot-generation-columns">
      <GenerationNumberField label="너비 (px)" value={value.img_w} min={256} max={4096} snapToStep={16} onChange={next => update("img_w", next)} />
      <GenerationNumberField label="높이 (px)" value={value.img_h} min={256} max={4096} snapToStep={16} onChange={next => update("img_h", next)} />
    </fieldset>
    {!manga && <p className="field-help">너비·높이는 256~4096px이며, 입력을 마치면 가장 가까운 16의 배수로 맞춰집니다.</p>}
    {manga && <p className="field-help">만화 출력에서는 아래 만화 이미지 크기·시드 설정을 사용합니다. 모델이 해상도 목록에서 고른 페이지 크기에 확대 배율을 곱해 그립니다.</p>}
    <fieldset className="illust-bot-generation-seed" disabled={!capabilities.local_advanced}>
      <GenerationNumberField label="시드" value={value.seed} min={-1} onChange={next => update("seed", next)} />
      <button type="button" className="button" onClick={() => update("seed", Math.floor(Math.random() * 4294967295))}><Dice5 size={14} />무작위 값</button>
      <button type="button" className="button" onClick={() => update("seed", -1)}><RotateCcw size={14} />매번 무작위</button>
    </fieldset>
    <p className="field-help">시드가 -1이면 생성할 때마다 무작위 값을 사용합니다.</p>
  </fieldset>;
}

type MangaPageSize = { value: string; width: number; height: number; label?: string };
// Keep the section usable even when an older server does not list the sizes.
const defaultMangaPageSizes: MangaPageSize[] = [
  { value: "1152x1728", width: 1152, height: 1728 },
  { value: "1408x1408", width: 1408, height: 1408 },
  { value: "1056x1888", width: 1056, height: 1888 },
  { value: "1536x1152", width: 1536, height: 1152 },
  { value: "1728x1152", width: 1728, height: 1152 },
];

function MangaSizeSeedFields({ value, pageSizes, readOnly, locked = false, update }: {
  value: IllustrationData; pageSizes?: MangaPageSize[]; readOnly: boolean; locked?: boolean; update: (field: string, value: unknown) => void;
}) {
  const shown = scaleValue(value.manga_scale);
  const sizes = pageSizes?.length ? pageSizes : defaultMangaPageSizes;
  const saved = value.manga_page_sizes === undefined || value.manga_page_sizes === null
    ? sizes.map(item => item.value).join(",") : String(value.manga_page_sizes);
  const selected = new Set(saved.split(",").map(item => item.trim()).filter(Boolean));
  // Show the size the page will actually be drawn at for the current scale.
  const scaled = (width: number, height: number) => [width, height]
    .map(size => Math.max(16, Math.round(size * shown / 16) * 16)).join("×");
  const toggle = (token: string, checked: boolean) => {
    const next = new Set(selected);
    if (checked) next.add(token); else next.delete(token);
    // The model needs at least one page size to choose from.
    if (!next.size) return;
    update("manga_page_sizes", sizes.filter(item => next.has(item.value)).map(item => item.value).join(","));
  };
  return <fieldset className={`illust-bot-generation-section illust-form${locked ? " illust-locked" : ""}`} data-group="manga-image-size-seed" disabled={readOnly || locked} aria-disabled={readOnly || locked || undefined}>
    <legend>만화 이미지 크기·시드{locked && <span>일반 삽화에서는 잠김</span>}</legend>
    <label className="field illust-bot-generation-range"><span>확대 배율<output>{scaleLabel(shown)}</output></span>
      <input type="range" aria-label="만화 확대 배율" min={0.2} max={2} step={0.1} value={shown} onChange={event => update("manga_scale", Number(event.target.value))} /></label>
    <p className="field-help">x1이 원본(기본값)입니다. x0.2~x2 범위에서 0.1 단위로 조절하며, 모델이 고른 페이지 해상도에 배율을 곱한 뒤 16배수 정수로 맞춰 그립니다.</p>
    <fieldset className="illust-bot-generation-sizes">
      <legend>해상도 목록</legend>
      {sizes.map(item => <label className="illust-check" key={item.value}>
        <input type="checkbox" checked={selected.has(item.value)} onChange={event => toggle(item.value, event.target.checked)} />
        <span>{item.label || `${item.width}×${item.height}`}{shown !== 1 && <small> → {scaled(item.width, item.height)}</small>}</span>
      </label>)}
    </fieldset>
    <p className="field-help">모델은 체크한 해상도 중에서 페이지 크기를 고릅니다. 여러 개를 중복 선택할 수 있으며, 최소 하나는 선택해야 합니다.</p>
    <fieldset className="illust-bot-generation-seed">
      <GenerationNumberField label="시드" value={value.manga_seed} min={-1} onChange={next => update("manga_seed", next)} />
      <button type="button" className="button" onClick={() => update("manga_seed", Math.floor(Math.random() * 4294967295))}><Dice5 size={14} />무작위 값</button>
      <button type="button" className="button" onClick={() => update("manga_seed", -1)}><RotateCcw size={14} />매번 무작위</button>
    </fieldset>
    <p className="field-help">시드가 -1이면 페이지를 그릴 때마다 무작위 값을 사용합니다.</p>
  </fieldset>;
}

function GenerationPresetEditor({ field, selected, readOnly, onClose, onSaved, onBusyChange }: {
  field: typeof presetFields[number]; selected: string; readOnly: boolean; onClose: () => void; onSaved: (name: string) => void; onBusyChange: (busy: boolean) => void;
}) {
  const [catalog, setCatalog] = useState<PresetCatalog | null>(null);
  const [revision, setRevision] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState(selected);
  const [text, setText] = useState("");
  const [creating, setCreating] = useState(!selected);
  const [discard, setDiscard] = useState(false);
  const initial = useRef("");
  useEffect(() => {
    const controller = new AbortController();
    void presetRequest<PresetCatalog>("", undefined, undefined, controller.signal).then(data => {
      setCatalog(data); setError("");
      const value = data.active[field.category]?.[selected] ?? data.builtin[field.category]?.[selected];
      initial.current = value === undefined ? "" : valueText(value); setText(initial.current);
      if (selected && value === undefined) setCreating(true);
    }).catch(cause => { if (!controller.signal.aborted) setError(cause.message); });
    return () => controller.abort();
  }, [field.category, selected, revision]);
  useEffect(() => { onBusyChange(busy); return () => onBusyChange(false); }, [busy, onBusyChange]);
  const builtin = !!catalog && !creating && !(selected in (catalog.active[field.category] || {})) && selected in (catalog.builtin[field.category] || {});
  const dirty = text !== initial.current || name !== selected;
  const close = () => { if (dirty) setDiscard(true); else onClose(); };
  return <Modal title={`${field.label} 태그 편집`} onClose={close} closeDisabled={busy} backgroundable={false}>
    {!catalog ? <IllustrationLoading error={error} retry={() => setRevision(value => value + 1)} /> : <form className="illust-form" onSubmit={event => { event.preventDefault(); if (busy || readOnly || builtin) return; setBusy(true); setError("");
      void presetRequest<{ name: string }>("save", { category: field.category, name, tags_text: text, scope: "local", original_scope: "local", operation: creating ? "create" : "update", original_name: creating ? "" : selected, target_state: "active" }, catalog.revision)
        .then(result => onSaved(result.name)).catch(cause => setError(cause.message)).finally(() => setBusy(false));
    }}>
      <fieldset className="illust-form" disabled={readOnly || busy}>
        <div className="illust-actions">{builtin && <Badge>배포자료 · 읽기 전용</Badge>}<button type="button" className="button" onClick={() => { setCreating(true); setName(""); setText(""); }}>새 프리셋</button><button type="button" className="button" disabled={!text} onClick={() => { setCreating(true); let next = `${name || field.label} (복사)`; let index = 2; while (next in (catalog.active[field.category] || {}) || next in (catalog.builtin[field.category] || {}) || next in (catalog.hidden[field.category] || {})) next = `${name || field.label} (복사 ${index++})`; setName(next); }}><Copy size={14} />복제</button></div>
        <label className="field">프리셋 이름<input required value={name} readOnly={!creating} onChange={event => setName(event.target.value)} /></label>
        <TagTextarea label="태그" value={text} onChange={setText} readOnly={readOnly || builtin} disabled={busy} rows={10} />
        <p className="field-help">{creating ? "저장한 프리셋을 현재 설정에 적용합니다." : builtin ? "복제해서 사용자 프리셋으로 편집할 수 있습니다." : "태그 변경은 이 프리셋을 사용하는 다른 설정에도 적용됩니다."}</p>
        {error && <p className="notice error" role="alert">{error}</p>}
        <div className="illust-actions"><button className="button primary" disabled={builtin || !name.trim()}>{busy ? "저장 중…" : "저장 및 적용"}</button><button type="button" className="button" onClick={close}>닫기</button></div>
      </fieldset>
    </form>}
    {discard && <Modal title="태그 편집 취소" onClose={() => setDiscard(false)} backgroundable={false}><div className="illust-form"><p>저장하지 않은 태그 변경을 버릴까요?</p><div className="illust-actions"><button className="button" onClick={() => setDiscard(false)}>계속 편집</button><button className="button danger" onClick={onClose}>변경 버리기</button></div></div></Modal>}
  </Modal>;
}
