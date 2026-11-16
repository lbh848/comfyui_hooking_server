import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Crop, Play, RefreshCw } from "lucide-react";
import { Badge, Modal, SettingsTabs } from "./components";
import { IllustTagFilterEditor, tagFilterStepLabel, type TagFilterStep, type TagFilterProfiles } from "./IllustTagFilterEditor";
import { TagTextarea } from "./TagTextarea";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { query, useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";
import { isFinishedQueueItem } from "./workQueue";
import { IllustPreparationFaceCrop } from "./IllustPreparationFaceCrop";
import "./illust-data-preparation.css";

export const dataPreparationTitle = (botName: string, label = "삽화 데이터 준비") => `${botName} · ${label}`;
const steps = ["캐릭터 선택", "기능 선택", "세부 설정", "최종 확인"];
const statusLabels: Record<string, string> = { pending: "대기", running: "진행 중", paused: "확인 대기", completed: "완료", skipped: "건너뜀", cancelled: "중단됨", warning: "확인 필요", failed: "실패" };

export function useDataPreparationRefresh(onChanged: (requery?: boolean) => void) {
  const runningBots = useRef(new Set<string>());
  useEffect(() => {
    let preparationRunning = false;
    let refreshPending = false;
    let timer: number | undefined;
    // Completion is published before queue status. Defer the screen refresh until
    // that status is received, including runs opened in another page or tab.
    const refresh = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        const requery = !preparationRunning && !runningBots.current.size;
        // An open preparation dialog already refreshes at its stage boundary.
        refreshPending = !requery && !runningBots.current.size;
        onChanged(requery);
      }, 0);
    };
    const receiveStatus = (event: Event) => {
      const items: IllustrationData[] = (event as CustomEvent<IllustrationData>).detail.items || [];
      preparationRunning = items.some(item => item.params?.one_click_run_id && !isFinishedQueueItem(item));
      if (refreshPending && !preparationRunning && !runningBots.current.size) refresh();
    };
    window.addEventListener("illustration-completed", refresh);
    window.addEventListener("illustration-queue-status", receiveStatus);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("illustration-completed", refresh);
      window.removeEventListener("illustration-queue-status", receiveStatus);
    };
  }, [onChanged]);
  return useCallback((botName: string, running: boolean) => {
    if (running) runningBots.current.add(botName);
    else runningBots.current.delete(botName);
  }, []);
}

function PreparationIssues({ stage, disabled, running, onRetry, onCrop }: {
  stage: IllustrationData; disabled: boolean; running: boolean; onRetry: (id: string) => void; onCrop: (item: IllustrationData) => void;
}) {
  const failures: IllustrationData[] = stage.failures || [];
  const known = new Set(failures.map(failure => failure.issue));
  const legacy = (stage.issues || []).filter((issue: string) => !known.has(issue)).map((issue: string, index: number) => {
    const match = issue.match(/^LLM 얼굴\/눈 태그:\s*(.+)\/(.+)\[(\d+)\]:\s*([\s\S]+)$/);
    return { id: `legacy-${index}`, status: "failed", message: match?.[4] || issue, character: match?.[2], visual_card_index: match?.[3], retryable: false };
  });
  const items: IllustrationData[] = [...failures, ...legacy];
  if (!items.length) return null;
  const resolved = items.filter(item => item.status === "resolved").length;
  return <details className="illust-preparation-issues" open><summary>확인할 항목 {items.length - resolved}개{resolved > 0 && ` · 처리 완료 ${resolved}개`}</summary>
    <ul>{items.map(item => {
      const target = [item.character, item.visual_card_index && `${item.visual_card_index}번 프로필`, item.visual_card_label].filter(Boolean).join(" · ");
      const parseError = String(item.message).includes("얼굴/눈 태그 JSON 파싱 실패");
      const croppable = !!item.character && (item.stage === "data_patch" || item.filename === "_face_image.webp");
      const manual = item.resolution === "manual_crop";
      return <li key={item.id} className={`illust-preparation-issue ${item.status}`}>
        <div className="illust-preparation-issue-heading"><strong>{target || "단계 실행 확인 필요"}</strong><Badge tone={item.status === "resolved" ? "green" : "rose"}>{item.status === "resolved" ? manual ? "직접 저장 완료" : "처리 완료" : item.status === "retrying" ? "재시도 중" : "실패 / 확인 필요"}</Badge></div>
        {item.filename && <p className="field-help">대상 이미지: <code>{item.filename}</code></p>}
        {item.status === "resolved" ? <p>{manual ? "직접 자른 얼굴을 이 프로필의 FACE 이미지로 저장했습니다." : "이 항목을 다시 처리해 저장했습니다."}</p> : <>
          <p>{parseError ? "LLM 응답을 얼굴·눈 태그 JSON으로 읽지 못해 정제 결과를 저장하지 못했습니다." : item.message}</p>
          {parseError && <details><summary>오류 상세</summary><pre>{item.message}</pre></details>}
          {(croppable || item.retryable) && <div className="illust-preparation-issue-actions">
            {croppable && <button type="button" className="button secondary" disabled={disabled || running} aria-label={`${target} ${stage.label} 얼굴 직접 자르기`} onClick={() => onCrop(item)}><Crop size={13} />얼굴 직접 자르기</button>}
            {item.retryable && <button type="button" className="button secondary" disabled={disabled || running || item.status === "retrying"} aria-label={`${target} ${stage.label} 재시도`} onClick={() => onRetry(item.id)}><RefreshCw size={13} />{item.status === "retrying" ? "재시도 중…" : "이 항목만 재시도"}</button>}
            <span className="field-help">{running ? "현재 실행이 끝나면 다시 처리할 수 있습니다." : croppable ? "대표 이미지에서 얼굴을 직접 잘라 이 항목의 FACE 이미지로 저장합니다." : "이 프로필의 실패한 작업만 다시 실행합니다."}</span>
          </div>}
        </>}
      </li>;
    })}</ul>
  </details>;
}

export function IllustDataPreparation({ botName, characters, readOnly, initialBackgrounded = false, label = "삽화 데이터 준비", onClose, onChanged, onRunningChange }: {
  botName: string; characters: IllustrationData[]; readOnly: boolean; initialBackgrounded?: boolean; label?: string; onClose: () => void; onChanged: () => void;
  onRunningChange?: (botName: string, running: boolean) => void;
}) {
  const api = useBotApi();
  const [data, setData] = useState<IllustrationData | null>(null);
  const [loadError, setLoadError] = useState("");
  const [revision, setRevision] = useState(0);
  const [step, setStep] = useState(0);
  const [view, setView] = useState("batch");
  const [stageKey, setStageKey] = useState("");
  const tabsId = useId();
  const [selected, setSelected] = useState<string[]>([]);
  const [enabled, setEnabled] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [negativePreset, setNegativePreset] = useState("");
  const [negativeTags, setNegativeTags] = useState("");
  const [cropTop, setCropTop] = useState("2.5");
  const [cropBottom, setCropBottom] = useState("1");
  const [faceFilterProfile, setFaceFilterProfile] = useState("");
  const [filterDraft, setFilterDraft] = useState<TagFilterStep[]>([]);
  const [tagFilterProfile, setTagFilterProfile] = useState("");
  const [tagFilterDraft, setTagFilterDraft] = useState<TagFilterStep[]>([]);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [run, setRun] = useState<IllustrationData | null>(null);
  const [pollError, setPollError] = useState("");
  const [stopping, setStopping] = useState(false);
  const [confirmStop, setConfirmStop] = useState(false);
  const [cropIssue, setCropIssue] = useState<IllustrationData | null>(null);
  const action = useIllustrationAction();
  const notified = useRef("");
  const running = run?.status === "running";
  const paused = run?.status === "paused";
  const changedStages = (run?.stages || []).filter((stage: IllustrationData) => (!running && stage.done > 0) || ["completed", "warning", "failed", "cancelled"].includes(stage.status));
  const resultRevision = run && (!running || changedStages.length)
    ? JSON.stringify([run.id, run.status, changedStages.map((stage: IllustrationData) => [stage.key, stage.status, stage.done])]) : "";
  useEffect(() => {
    onRunningChange?.(botName, running);
    return () => onRunningChange?.(botName, false);
  }, [botName, running, onRunningChange]);
  const disabled = readOnly || action.busy || settingsBusy;
  const rows = characters.map(character => {
    const profiles: IllustrationData[] = character.card_data?.character?.profiles || [];
    const targets = profiles.filter(profile => profile.render_overrides?.rep_images?.length);
    return { name: character.name as string, profiles, targets, count: targets.reduce((sum, profile) => sum + profile.render_overrides.rep_images.length, 0), generated: character.npc_generated === true };
  });
  const eligible = rows.filter(row => row.targets.length > 0);
  const selectedRows = rows.filter(row => selected.includes(row.name));
  const profileCount = selectedRows.reduce((sum, row) => sum + row.targets.length, 0);
  const needsNegative = enabled.includes("negative") || enabled.includes("face_negative");
  const needsCrop = enabled.includes("data_patch");
  const needsFilter = enabled.includes("face_filter");
  const needsTagFilter = enabled.includes("tag_filter");
  const filterSteps = data?.filters.profiles[faceFilterProfile] || [];
  const tagFilterSteps = data?.filters.profiles[tagFilterProfile] || [];
  const faceFilterDirty = JSON.stringify(filterDraft) !== JSON.stringify(filterSteps);
  const tagFilterDirty = JSON.stringify(tagFilterDraft) !== JSON.stringify(tagFilterSteps);
  const filterDirty = faceFilterDirty || tagFilterDirty;
  const invalidCrop = [cropTop, cropBottom].some(value => !value || !Number.isFinite(Number(value)) || Number(value) < .1 || Number(value) > 10);
  function invalidFor(stages: string[]) {
    return !selected.length ? "캐릭터를 한 명 이상 선택하세요." : !stages.length ? "실행할 기능을 선택하세요." : stages.some(key => ["negative", "face_negative"].includes(key)) && !negativeTags.trim() ? "부정 태그를 입력하세요." : stages.includes("data_patch") && invalidCrop ? "FACE 크롭 값은 0.1~10 사이여야 합니다." : filterDirty ? "변경한 필터 규칙을 저장하거나 변경 취소를 눌러 주세요." : "";
  }
  const invalid = invalidFor(enabled);
  const activeStage = data?.stages.find((stage: IllustrationData) => stage.key === stageKey);
  const filterTarget = stageKey === "tag_filter" ? "representative" : "face";
  const activeFilterSteps = filterTarget === "face" ? filterSteps : tagFilterSteps;

  function selectFilter(name: string, target: "representative" | "face" = "face") {
    if (target === "face") { setFaceFilterProfile(name); setFilterDraft(data?.filters.profiles[name] || []); }
    else { setTagFilterProfile(name); setTagFilterDraft(data?.filters.profiles[name] || []); }
  }
  function filtersSaved(filters: TagFilterProfiles, name: string, target: "representative" | "face") {
    const face = target === "face" ? name : faceFilterProfile && !(faceFilterProfile in filters.profiles) ? filters.active_profile : faceFilterProfile;
    const representative = target === "representative" ? name : tagFilterProfile && !(tagFilterProfile in filters.profiles) ? filters.active_profile : tagFilterProfile;
    if (target === "face" || !faceFilterDirty) { setFaceFilterProfile(face); setFilterDraft(filters.profiles[face] || []); }
    if (target === "representative" || !tagFilterDirty) { setTagFilterProfile(representative); setTagFilterDraft(filters.profiles[representative] || []); }
    setData(current => current && { ...current, filters });
  }
  function close() { if (filterDirty) setConfirmDiscard(true); else onClose(); }
  function openFilterSettings(target: "representative" | "face") { setView("stages"); setStageKey(target === "face" ? "face_filter" : "tag_filter"); }
  function filterSettings(target: "representative" | "face") {
    const name = target === "face" ? faceFilterProfile : tagFilterProfile;
    const rules: TagFilterStep[] = target === "face" ? filterSteps : tagFilterSteps;
    const label = target === "face" ? "얼굴 태그 필터" : "대표 이미지 태그 필터";
    return <section className="illust-preparation-setting illust-form"><label className="field">{label} 프로필<select value={name} disabled={target === "face" ? faceFilterDirty : tagFilterDirty} onChange={event => selectFilter(event.target.value, target)}><option value="">사용 안 함</option>{Object.entries(data?.filters.profiles || {}).sort(([a], [b]) => a.localeCompare(b, "ko", { numeric: true })).map(([profile, rules]) => <option key={profile} value={profile}>{profile} ({(rules as unknown[]).length ? `${(rules as unknown[]).length}단계` : "규칙 없음"})</option>)}</select></label>
      {rules.length > 0 ? <ol className="illust-preparation-review">{rules.map((rule, index) => <li key={index}>{tagFilterStepLabel(rule)}</li>)}</ol> : <p className="field-help">{name ? `‘${name}’ 프로필에 저장된 규칙이 없어 ${label}링을 건너뜁니다.` : `${label}링을 사용하지 않습니다.`}</p>}
      {(target === "face" ? faceFilterDirty : tagFilterDirty) && <p className="field-error">단계별 설정에서 편집한 규칙을 저장해 주세요.</p>}
      <div><button type="button" className="button secondary" onClick={() => openFilterSettings(target)}>{label} 규칙 설정</button></div>
    </section>;
  }

  useEffect(() => {
    const controller = new AbortController();
    setLoadError("");
    void Promise.all([
      api.request(`bot_mode/data_preparation?${query({ bot: botName })}`, undefined, controller.signal),
      api.request(`bot_mode/patch_settings?${query({ bot: botName })}`, undefined, controller.signal),
      api.request("asset_mode/tags", undefined, controller.signal),
      api.request("bot_mode/tag_filter_profiles", undefined, controller.signal),
    ]).then(([preparation, patch, tags, filters]) => {
      if (controller.signal.aborted) return;
      const presets = tags.negative_presets || {};
      const first = Object.keys(presets)[0] || "";
      setData({ stages: preparation.stages, patch, presets, filters });
      setSelected(eligible.map(row => row.name)); setEnabled(preparation.stages.map((stage: IllustrationData) => stage.key));
      setCropTop(String(patch.face_crop_top ?? 2.5)); setCropBottom(String(patch.face_crop_bottom ?? 1));
      setNegativePreset(first); setNegativeTags((presets[first] || []).join(", "));
      const profile = filters.active_profile || Object.keys(filters.profiles || {})[0] || "";
      setFaceFilterProfile(profile); setFilterDraft(filters.profiles[profile] || []);
      setTagFilterProfile(profile); setTagFilterDraft(filters.profiles[profile] || []);
      setRun(preparation.run);
    }).catch(error => { if (!controller.signal.aborted) setLoadError(error.message); });
    return () => controller.abort();
  }, [botName, revision]);
  useEffect(() => {
    if (!running) return;
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const result = await api.request(`bot_mode/data_preparation?${query({ bot: botName })}`, undefined, controller.signal);
        if (!controller.signal.aborted) { setRun(result.run); setPollError(""); }
      } catch (error) { if (!controller.signal.aborted) setPollError((error as Error).message); }
      finally { if (!controller.signal.aborted) timer = window.setTimeout(poll, 1500); }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [botName, running]);
  useEffect(() => {
    // Progress stays live through polling. Refresh the catalog at stage boundaries
    // and on pause/stop, rather than reloading every character after each image.
    if (resultRevision && notified.current !== resultRevision) { notified.current = resultRevision; onChanged(); }
  }, [resultRevision, onChanged]);

  async function selectChanged() {
    const result = await api.request(`bot_mode/utility_preview?${query({ bot: botName })}`);
    const faces = new Map<string, IllustrationData>((result.images || []).map((image: IllustrationData) => [`${image.character}:${image.visual_card_id}`, image]));
    setSelected(eligible.filter(row => row.targets.some(profile => {
      const face = faces.get(`${row.name}:${profile.id}`);
      return !face || face.face_matches_rep === false;
    })).map(row => row.name));
  }
  function next() {
    if (step === 0 && !selected.length || step === 1 && !enabled.length || step === 2 && invalid) return;
    setStep(value => Math.min(3, value + 1));
  }
  async function start(stages = enabled) {
    if (disabled || invalidFor(stages)) return;
    await action.run(async () => {
      const result = await api.request("bot_mode/data_preparation", { action: "start", bot_name: botName, characters: selected, stages, negative_tags: negativeTags, crop_top: invalidCrop ? data?.patch.face_crop_top ?? 2.5 : Number(cropTop), crop_bottom: invalidCrop ? data?.patch.face_crop_bottom ?? 1 : Number(cropBottom), face_filter_profile: faceFilterProfile, tag_filter_profile: tagFilterProfile });
      setRun(result.run);
      setStageKey("");
    }, "");
  }
  async function stop() {
    if (!run || stopping) return;
    setStopping(true);
    await action.run(async () => {
      const result = await api.request("bot_mode/data_preparation", { action: "stop", bot_name: botName, run_id: run.id });
      setRun(result.run); setConfirmStop(false);
    }, "");
    setStopping(false);
  }
  async function retryIssue(id: string) {
    if (!run || running || disabled) return;
    await action.run(async () => {
      const result = await api.request("bot_mode/data_preparation", { action: "retry", bot_name: botName, run_id: run.id, issue_ids: [id] });
      setRun(result.run);
    }, "");
  }
  async function applyCroppedFace(dataUrl: string) {
    if (!run || !cropIssue) throw new Error("얼굴을 저장할 대상이 없습니다.");
    const result = await api.request("bot_mode/data_preparation", {
      action: "face_crop", bot_name: botName, run_id: run.id, issue_id: cropIssue.id, image: dataUrl,
    });
    setRun(result.run);
    setCropIssue(null);
  }
  async function continueRemaining() {
    if (!run || !paused || disabled) return;
    await action.run(async () => {
      const result = await api.request("bot_mode/data_preparation", { action: "continue", bot_name: botName, run_id: run.id });
      setRun(result.run);
    }, "");
  }
  const negativeSettings = data && <div className="illust-form"><label className="field">프리셋<select value={negativePreset} onChange={event => { setNegativePreset(event.target.value); if (event.target.value) setNegativeTags(data.presets[event.target.value].join(", ")); }}><option value="">직접 입력</option>{Object.entries(data.presets).sort(([a], [b]) => a.localeCompare(b, "ko", { numeric: true })).map(([name, tags]) => <option key={name} value={name}>{name} ({(tags as unknown[]).length}개)</option>)}</select></label><TagTextarea label="적용할 부정 태그" ariaLabel="적용할 부정 태그" value={negativeTags} onChange={setNegativeTags} disabled={disabled} rows={5} /><p className="field-help">대표 이미지와 FACE 이미지에 적용할 부정 태그입니다.</p></div>;
  const cropSettings = <div className="illust-fields"><label className="field">CROP_TOP<input type="number" min="0.1" max="10" step="0.1" value={cropTop} onChange={event => setCropTop(event.target.value)} /></label><label className="field">CROP_BOTTOM<input type="number" min="0.1" max="10" step="0.1" value={cropBottom} onChange={event => setCropBottom(event.target.value)} /></label></div>;
  const visibleCharacters = rows.filter(row => row.name.toLowerCase().includes(search.toLowerCase()));
  const userCharacters = visibleCharacters.filter(row => !row.generated);
  const generatedCharacters = visibleCharacters.filter(row => row.generated);
  const renderCharacter = (row: typeof rows[number]) => {
    const filename = row.targets[0]?.render_overrides.rep_images[0];
    return <label className={`illust-preparation-character ${selected.includes(row.name) ? "selected" : ""}`} key={row.name}><input type="checkbox" checked={selected.includes(row.name)} disabled={disabled || !row.targets.length} onChange={event => setSelected(current => event.target.checked ? [...current, row.name] : current.filter(name => name !== row.name))} />{filename && <img alt="" src={api.media(`/api/bot_mode/image/${encodeURIComponent(botName)}/${encodeURIComponent(row.name)}/${encodeURIComponent(filename)}`)} loading="lazy" />}<span><strong>{row.name}</strong><small>{row.targets.length ? `프로필 ${row.targets.length}/${row.profiles.length}개 · 대표 ${row.count}장` : "대표 이미지 미설정"}</small></span></label>;
  };
  const characterSelection = <div className="illust-form"><div className="illust-actions"><h3>캐릭터 선택 <Badge>{selected.length}명 · 프로필 {profileCount}개</Badge></h3><button type="button" className="button" disabled={disabled} onClick={() => setSelected(eligible.map(row => row.name))}>전체 선택</button><button type="button" className="button" disabled={disabled} onClick={() => setSelected([])}>전체 해제</button><button type="button" className="button secondary" disabled={disabled} onClick={() => void action.run(selectChanged, "")}><RefreshCw size={13} />FACE 없음 / 대표와 다름</button></div>
    <label className="field">캐릭터 이름 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} /></label>
    <div className="illust-preparation-characters">{userCharacters.map(renderCharacter)}{userCharacters.length > 0 && generatedCharacters.length > 0 && <div className="illust-character-group-divider" role="separator" aria-label="생성된 캐릭터 구분선"><span>생성된 캐릭터</span></div>}{generatedCharacters.map(renderCharacter)}</div>{!visibleCharacters.length && <p className="field-help">검색된 캐릭터가 없습니다.</p>}
  </div>;
  return <Modal title={dataPreparationTitle(botName, label)} initialBackgrounded={initialBackgrounded} onClose={close} closeDisabled={running || action.busy || settingsBusy} working={running || action.busy || settingsBusy} className="illust-preparation-modal">
    {!data ? <IllustrationLoading error={loadError} retry={() => setRevision(value => value + 1)} /> : run ? <div className="illust-form illust-preparation-run">
      <div className="illust-preparation-run-heading"><div><h3>{run.stop_requested && running ? "안전 중단 중…" : `${label} · ${statusLabels[run.status]}`}</h3><p className="field-help">{run.characters.length}명 · {run.characters.join(", ")}</p></div><Badge>{run.stages.filter((stage: IllustrationData) => ["completed", "warning", "failed"].includes(stage.status)).length} / {run.stages.filter((stage: IllustrationData) => stage.status !== "skipped").length}</Badge></div>
      <div className="illust-preparation-progress" aria-live="polite">{run.stages.map((stage: IllustrationData, i: number) => <article key={stage.key} className={`illust-preparation-stage ${stage.status}`}>
        <span className="illust-preparation-number">{stage.status === "completed" ? <Check size={15} /> : i + 1}</span><div><div className="illust-actions"><strong>{stage.label}</strong><Badge>{statusLabels[stage.status]}</Badge></div><p className="field-help">{stage.detail || "앞 단계가 끝나면 실행합니다."}</p>{stage.total > 0 && <progress aria-label={`${stage.label} 진행률`} max={stage.total} value={stage.done} />}<PreparationIssues stage={stage} disabled={disabled} running={running} onRetry={id => void retryIssue(id)} onCrop={setCropIssue} /></div>
      </article>)}</div>
      <IllustrationFeedback error={pollError || run.error || action.error} />
      {running && <p className="field-help">백그라운드로 보내도 작업은 계속됩니다. 안전 중단은 현재 처리 중인 작업을 마친 뒤 이 실행의 대기 작업과 나머지 단계를 취소합니다.</p>}
      {paused && <p className="field-help">실패한 항목의 얼굴을 직접 잘라 저장하면 나머지 단계를 이어서 진행합니다. 자르지 않은 항목이 남아 있어도 ‘나머지 단계 계속’으로 진행할 수 있습니다.</p>}
      <div className="modal-actions">{running ? <button type="button" className="button danger" disabled={stopping || run.stop_requested} onClick={() => setConfirmStop(true)}>{run.stop_requested ? "중단 처리 중…" : "안전 중단"}</button> : <>{paused && <button type="button" className="button primary" disabled={action.busy} onClick={() => void continueRemaining()}><Play size={14} />{action.busy ? "시작 중…" : "나머지 단계 계속"}</button>}<button type="button" className="button secondary" disabled={readOnly || action.busy} onClick={() => { setRun(null); setStep(0); }}>새 원클릭 설정</button><button type="button" className="button primary" onClick={onClose}>확인</button></>}</div>
    </div> : <div className="illust-form">
      <SettingsTabs id={tabsId} label={`${label} 설정`} values={{ batch: "일괄 설정", stages: "단계별 설정" }} value={view} onChange={setView} />
      <div role="tabpanel" id={`${tabsId}-panel-batch`} aria-labelledby={`${tabsId}-tab-batch`} hidden={view !== "batch"} className="illust-preparation-panel illust-preparation-wizard">
      <ol className="illust-preparation-steps" aria-label="원클릭 설정 단계">{steps.map((label, i) => <li key={label} aria-current={step === i ? "step" : undefined}><span>{i + 1}</span>{label}</li>)}</ol>
      <p className="field-help">저장된 카드의 대표 이미지를 기준으로 선택한 기능을 순서대로 실행합니다.</p>
      <div className="illust-preparation-page">
        {step === 0 && characterSelection}
        {step === 1 && <div className="illust-form"><div className="illust-actions"><h3>실행할 기능 선택 <Badge>{enabled.length} / {data.stages.length}</Badge></h3><button type="button" className="button" onClick={() => setEnabled(data.stages.map((stage: IllustrationData) => stage.key))}>모두 사용</button><button type="button" className="button" onClick={() => setEnabled([])}>모두 해제</button></div><div className="illust-preparation-functions">{data.stages.map((stage: IllustrationData, i: number) => <label key={stage.key} className={`illust-preparation-function ${enabled.includes(stage.key) ? "selected" : ""}`}><input type="checkbox" checked={enabled.includes(stage.key)} disabled={disabled} onChange={event => setEnabled(current => event.target.checked ? [...current, stage.key] : current.filter(key => key !== stage.key))} /><span className="illust-preparation-number">{i + 1}</span><span><strong>{stage.label}</strong><small>{stage.description}</small></span></label>)}</div></div>}
        {step === 2 && <fieldset className="illust-form" disabled={disabled}>
          {needsTagFilter && filterSettings("representative")}
          {needsFilter && filterSettings("face")}
          {needsNegative && <section className="illust-preparation-setting illust-form"><h3>부정 태그 설정</h3>{negativeSettings}</section>}
          {needsCrop && <section className="illust-preparation-setting illust-form"><h3>FACE 크롭 설정</h3>{cropSettings}<p className="field-help">이번 실행에만 적용됩니다. 단계별 설정에서 봇의 기본값으로 저장할 수 있습니다.</p></section>}
          {!needsNegative && !needsCrop && !needsFilter && !needsTagFilter && <p className="field-help">선택한 기능에는 추가 설정이 필요하지 않습니다.</p>}
        </fieldset>}
        {step === 3 && <div className="illust-form"><h3>실행 내용 확인</h3><section className="illust-preparation-setting"><strong>캐릭터 {selected.length}명 · 프로필 {profileCount}개</strong><p>{selected.join(", ")}</p></section><section className="illust-preparation-setting"><h3>실행 순서</h3><ol className="illust-preparation-review">{data.stages.filter((stage: IllustrationData) => enabled.includes(stage.key)).map((stage: IllustrationData) => <li key={stage.key}>{stage.label}</li>)}</ol></section>{needsTagFilter && <p className="field-help">대표 이미지 태그 필터: {tagFilterProfile || "사용 안 함"} · {tagFilterSteps.length ? `${tagFilterSteps.length}단계` : "규칙이 없어 건너뜀"}</p>}{needsFilter && <p className="field-help">얼굴 태그 필터: {faceFilterProfile || "사용 안 함"} · {filterSteps.length ? `${filterSteps.length}단계` : "규칙이 없어 건너뜀"}</p>}{needsCrop && <p className="field-help">FACE 크롭: 위 {cropTop} / 아래 {cropBottom}</p>}{needsNegative && <section className="illust-preparation-setting"><h3>적용할 부정 태그</h3><p className="illust-text">{negativeTags}</p></section>}</div>}
      </div>
      <IllustrationFeedback error={action.error} />
      <footer className="illust-preparation-footer"><span className="field-help" role="status">{(step === 0 && !selected.length || step === 1 && !enabled.length || step >= 2) && invalid}</span><div className="illust-actions"><button type="button" className="button" disabled={action.busy || settingsBusy} onClick={close}>취소</button>{step > 0 && <button type="button" className="button secondary" disabled={action.busy} onClick={() => setStep(value => value - 1)}><ChevronLeft size={14} />이전</button>}{step < 3 ? <button type="button" className="button primary" disabled={disabled || step === 0 && !selected.length || step === 1 && !enabled.length || step === 2 && !!invalid} onClick={next}>다음<ChevronRight size={14} /></button> : <button type="button" className="button primary" disabled={disabled || !!invalid} onClick={() => void start()}><Play size={14} />{action.busy ? "등록 중…" : "원클릭 시작"}</button>}</div></footer>
      </div>
      <div role="tabpanel" id={`${tabsId}-panel-stages`} aria-labelledby={`${tabsId}-tab-stages`} hidden={view !== "stages"} className="illust-preparation-panel illust-form">
        <p className="field-help">작업을 선택해 설정 창을 열고 필요한 단계만 실행하세요. 캐릭터 선택과 설정값은 일괄 설정과 공유합니다.</p>
        <div className="illust-preparation-page illust-form">
          <div className="illust-preparation-targets"><div><strong>작업할 캐릭터</strong><span>{selected.length}명 · 프로필 {profileCount}개</span></div><button type="button" className="button secondary" onClick={() => setStageKey("characters")}>캐릭터 선택</button></div>
          <div className="illust-preparation-stage-grid">{data.stages.map((stage: IllustrationData, index: number) => <button type="button" className="illust-preparation-stage-button" aria-haspopup="dialog" key={stage.key} onClick={() => setStageKey(stage.key)}>
            <span className="illust-preparation-number">{index + 1}</span><span><strong>{stage.label}</strong><small>{stage.description}</small></span><ChevronRight size={14} />
          </button>)}</div>
        </div>
        <IllustrationFeedback error={action.error} message={action.message} />
        <footer className="illust-preparation-footer"><span className="field-help">{filterDirty ? "변경한 필터 규칙을 저장해 주세요." : `선택한 캐릭터 ${selected.length}명`}</span><div className="illust-actions"><button type="button" className="button" disabled={action.busy || settingsBusy} onClick={close}>닫기</button><button type="button" className="button secondary" onClick={() => setView("batch")}>일괄 설정으로 돌아가기</button></div></footer>
      </div>
    </div>}
    {data && !run && view === "stages" && stageKey === "characters" && <Modal title="작업할 캐릭터 선택" backgroundable={false} className="illust-preparation-detail-modal" onClose={() => setStageKey("")} closeDisabled={action.busy || settingsBusy}>
      <div className="illust-preparation-detail-body">{characterSelection}</div><footer className="illust-preparation-footer"><span className="field-help">{selected.length}명 · 프로필 {profileCount}개 선택</span><button type="button" className="button primary" disabled={action.busy || settingsBusy} onClick={() => setStageKey("")}>선택 완료</button></footer>
    </Modal>}
    {data && !run && view === "stages" && activeStage && <Modal title={activeStage.label} backgroundable={false} className="illust-preparation-detail-modal" onClose={() => setStageKey("")} closeDisabled={action.busy || settingsBusy}>
      <div className="illust-preparation-detail-body illust-form">
        <p className="field-help">{activeStage.description} · 선택한 캐릭터 {selected.length}명</p>
        {["tag_filter", "face_filter"].includes(stageKey) ? <IllustTagFilterEditor key={filterTarget} botName={botName} characters={selected} target={filterTarget} filters={data.filters} profile={filterTarget === "face" ? faceFilterProfile : tagFilterProfile} steps={filterTarget === "face" ? filterDraft : tagFilterDraft} disabled={disabled} onSelect={name => selectFilter(name, filterTarget)} onChange={filterTarget === "face" ? setFilterDraft : setTagFilterDraft} onSaved={(filters, name) => filtersSaved(filters, name, filterTarget)} onBusyChange={setSettingsBusy} /> : <fieldset className="illust-form" disabled={disabled}>
          {["negative", "face_negative"].includes(stageKey) && negativeSettings}
          {stageKey === "tag_analysis" && <p className="field-help">선택한 캐릭터의 모든 프로필에 저장된 대표 이미지 태그를 분석합니다.</p>}
          {stageKey === "llm_refine" && <p className="field-help">선택한 캐릭터의 FACE 이미지를 바탕으로 얼굴·눈 태그를 정제합니다. FACE 이미지가 없으면 데이터 패치를 먼저 실행하세요.</p>}
          {stageKey === "appearance_refine" && <p className="field-help">대표 이미지와 저장된 외모·복장 태그를 바탕으로 정제한 뒤 카드에 저장합니다. 기존 공통 정제 프롬프트를 사용하고, 캐릭터·카드별 일괄 잠금 설정을 따릅니다. FACE 이미지 없이도 실행할 수 있습니다.</p>}
          {stageKey === "face_tag_analysis" && <p className="field-help">선택한 캐릭터의 모든 프로필에 저장된 FACE 이미지 태그를 분석합니다.</p>}
          {stageKey === "face_negative" && <p className="field-help">선택한 캐릭터의 모든 프로필에 저장된 FACE 이미지에 위 부정 태그를 적용합니다.</p>}
          {stageKey === "data_patch" && <>{cropSettings}<p className="field-help">선택한 캐릭터의 데이터를 보정하고 프로필별 FACE 이미지를 생성합니다.</p><div><button type="button" className="button secondary" disabled={invalidCrop} onClick={() => void action.run(async () => {
            const settings = { ...data.patch, face_crop_top: Number(cropTop), face_crop_bottom: Number(cropBottom) };
            await api.request("bot_mode/patch_settings", { bot: botName, settings });
            const patch = await api.request(`bot_mode/patch_settings?${query({ bot: botName })}`);
            if (patch.face_crop_top !== settings.face_crop_top || patch.face_crop_bottom !== settings.face_crop_bottom) throw new Error("데이터 패치 설정 저장 결과를 확인할 수 없습니다.");
            setData(current => current && { ...current, patch });
          }, "데이터 패치 설정을 봇의 기본값으로 저장했습니다.")}>데이터 패치 설정 저장</button></div></>}
        </fieldset>}
        {invalidFor([stageKey]) && <p className="field-error">{invalidFor([stageKey])}</p>}
        {["tag_filter", "face_filter"].includes(stageKey) && !activeFilterSteps.length && <p className="field-help">실행하려면 필터 프로필에 규칙을 저장하세요.</p>}
        <IllustrationFeedback error={action.error} message={action.message} />
      </div>
      <footer className="illust-preparation-footer"><span className="field-help">선택한 캐릭터 {selected.length}명 · 이 단계만 실행</span><div className="illust-actions"><button type="button" className="button" disabled={action.busy || settingsBusy} onClick={() => setStageKey("")}>닫기</button><button type="button" className="button primary" disabled={disabled || !!invalidFor([stageKey]) || ["tag_filter", "face_filter"].includes(stageKey) && !activeFilterSteps.length} onClick={() => void start([stageKey])}><Play size={14} />{activeStage.label} 실행</button></div></footer>
    </Modal>}
    {confirmDiscard && <Modal title="필터 규칙 변경 취소" onClose={() => setConfirmDiscard(false)} backgroundable={false}><p>저장하지 않은 필터 규칙 변경을 취소하고 닫을까요?</p><div className="modal-actions"><button type="button" className="button" onClick={() => setConfirmDiscard(false)}>계속 편집</button><button type="button" className="button danger" onClick={onClose}>변경 취소하고 닫기</button></div></Modal>}
    {cropIssue && <IllustPreparationFaceCrop botName={botName} issue={cropIssue} onClose={() => setCropIssue(null)} onApply={applyCroppedFace} />}
    {confirmStop && <Modal title="안전 중단" onClose={() => setConfirmStop(false)} closeDisabled={stopping} backgroundable={false}><p>현재 처리 중인 작업은 마무리하고, 이 실행의 대기 작업과 나머지 단계를 취소합니다.</p><IllustrationFeedback error={action.error} /><div className="modal-actions"><button type="button" className="button" disabled={stopping} onClick={() => setConfirmStop(false)}>계속 실행</button><button type="button" className="button danger" disabled={stopping} onClick={() => void stop()}>안전 중단</button></div></Modal>}
  </Modal>;
}
