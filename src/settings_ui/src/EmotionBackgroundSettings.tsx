import { useEffect, useRef, useState, type FormEvent } from "react";
import { Blend, Copy, Globe, Images, LockKeyhole, LockKeyholeOpen, Pencil, Save, SlidersHorizontal, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { EmotionAutoBackgroundPresets } from "./EmotionAutoBackgroundPresets";
import { EmotionCompositionSettings } from "./EmotionCompositionSettings";
import { SearchableSelect } from "./SearchableSelect";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { emotionIllustrationRequest, useEmotionIllustrationResource } from "./emotionIllustration";
import { illustrationMedia, type IllustrationData } from "./illustration";
import { useWorkflowAutosave } from "./useWorkflowAutosave";
import { EmotionBotApiProvider } from "./BotApiScope";
import { IllustBotGenerationSettings } from "./IllustBotGenerationSettings";

/**
 * 감정 삽화 공방의 배경 설정.
 * 배경 생성은 수동 배경 모드와 자동 배경 모드로 나뉘며, 각 모드는 내부 탭으로
 * 관리한다. 수동 설정은 공방 공통 값, 자동 배경 봇은 별도 저장소를 사용한다.
 */
export const backgroundModes = {
  manual: "수동 배경 모드",
  auto: "자동 배경 모드",
};
export type BackgroundMode = keyof typeof backgroundModes;

// 사이드바 선택 표시처럼 "배경" 맥락이 이미 드러나는 곳에서 쓰는 짧은 이름.
export const backgroundModeShortLabels: Record<BackgroundMode, string> = {
  manual: "수동",
  auto: "자동",
};

const modeDescriptions: Record<BackgroundMode, string> = {
  manual: "배경 CSS 위에 백업 이미지를 놓고, 다시 그 위에 전면 CSS 레이어를 얹어 배경 구성을 확인합니다.",
  auto: "원문과 세계관 설정에 맞는 배경 카드를 재사용하고, 없는 배경은 새로 생성합니다.",
};

type PreviewImage = { name: string; url: string };
type PresetScope = "builtin" | "local";
type BackgroundPreset = {
  name: string;
  scope: PresetScope;
  bg_css: string;
  fg_css: string;
  preview_name: string;
  preview_url: string;
};
type PresetDialog = "save" | "duplicate" | "rename" | "delete" | "unlock";

const presetScopeLabels: Record<PresetScope, string> = {
  builtin: "배포 프리셋",
  local: "사용자 프리셋",
};

// 두 레이어는 미리보기 전용이다. 생성 결과에는 아직 사용하지 않는다.
function layerRules(selector: string, css: string): string {
  const body = css.trim();
  if (!body) return "";
  // 선택자를 직접 쓴 규칙은 그대로 두고, 선언만 입력하면 해당 레이어에 감싼다.
  return body.includes("{") ? body : `${selector}{${body}}`;
}

function formValue(bgCss: string, fgCss: string, preview: PreviewImage | null, mode: BackgroundMode = "manual", world = ""): string {
  return JSON.stringify([bgCss, fgCss, preview?.name || "", preview?.url || "", mode, world]);
}

export function EmotionBackgroundSettings({ mode, onModeChange, onPresetChange, botSelected }: {
  mode: BackgroundMode;
  onModeChange: (mode: BackgroundMode) => void;
  onPresetChange?: (name: string) => void;
  botSelected: string;
}) {
  const resource = useEmotionIllustrationResource("illustration/config");
  const presetsResource = useEmotionIllustrationResource("illustration/background-presets");
  const [bgCss, setBgCss] = useState("");
  const [fgCss, setFgCss] = useState("");
  const [world, setWorld] = useState("");
  const [worldOpen, setWorldOpen] = useState(false);
  const [generationBot, setGenerationBot] = useState("");
  const [generationDirty, setGenerationDirty] = useState(false);
  const [generationBusy, setGenerationBusy] = useState(false);
  const [discardGeneration, setDiscardGeneration] = useState(false);
  const [compositionOpen, setCompositionOpen] = useState(false);
  const [compositionDirty, setCompositionDirty] = useState(false);
  const [compositionBusy, setCompositionBusy] = useState(false);
  const [discardComposition, setDiscardComposition] = useState(false);
  const distribution = useEmotionIllustrationResource(generationBot ? "bot_mode/distribution" : null);
  const [preview, setPreview] = useState<PreviewImage | null>(null);
  const [picker, setPicker] = useState(false);
  const [ratio, setRatio] = useState(0.75);
  const [baseline, setBaseline] = useState(() => formValue("", "", null));
  const [presets, setPresets] = useState<BackgroundPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState("");
  const [presetDialog, setPresetDialog] = useState<PresetDialog | null>(null);
  const [presetName, setPresetName] = useState("");
  const [presetScope, setPresetScope] = useState<PresetScope>("local");
  const [presetBusy, setPresetBusy] = useState(false);
  const [presetError, setPresetError] = useState("");
  const [presetMessage, setPresetMessage] = useState("");
  // 배포 프리셋 편집은 이 세션에서 잠금을 해제한 동안에만 허용한다.
  const [unlockToken, setUnlockToken] = useState("");
  const form = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (!resource.data) return;
    const config = resource.data.config || {};
    const url = String(config.background_preview_url || "");
    const nextPreview = url
      ? { name: String(config.background_preview_name || ""), url }
      : null;
    const nextBgCss = String(config.background_bg_css || "");
    const nextFgCss = String(config.background_fg_css || "");
    const nextWorld = String(config.background_world || "");
    const nextMode: BackgroundMode = config.background_mode === "auto" ? "auto" : "manual";
    onModeChange(nextMode);
    setWorld(nextWorld);
    setBgCss(nextBgCss);
    setFgCss(nextFgCss);
    setPreview(nextPreview);
    setBaseline(formValue(nextBgCss, nextFgCss, nextPreview, nextMode, nextWorld));
  }, [resource.data, onModeChange]);

  useEffect(() => {
    if (!presetsResource.data) return;
    const next: BackgroundPreset[] = Array.isArray(presetsResource.data.presets)
      ? presetsResource.data.presets
      : [];
    setPresets(next);
    setSelectedPreset(current => next.some(item => item.name === current) ? current : "");
  }, [presetsResource.data]);

  // 화면을 떠나거나 토큰이 바뀌면 이전 세션의 배포 잠금을 되돌린다.
  useEffect(() => () => {
    if (unlockToken) void emotionIllustrationRequest("illustration/background-presets", {
      action: "lock", builtin_unlock_token: unlockToken,
    }).catch(() => {});
  }, [unlockToken]);

  // 선택한 백업 이미지의 원본 비율을 미리보기 박스에 반영한다.
  const previewUrl = preview ? illustrationMedia(preview.url) : "";
  useEffect(() => {
    if (!previewUrl) setRatio(0.75);
    if (!previewUrl) return;
    let active = true;
    const probe = new Image();
    probe.onload = () => {
      if (active && probe.naturalWidth > 0 && probe.naturalHeight > 0) {
        setRatio(probe.naturalWidth / probe.naturalHeight);
      }
    };
    probe.src = previewUrl;
    return () => { active = false; probe.onload = null; };
  }, [previewUrl]);

  const value = formValue(bgCss, fgCss, preview, mode, world);
  const dirty = value !== baseline;
  const autosave = useWorkflowAutosave({
    label: "감정 삽화 · 배경 설정", value, dirty, form, paused: !resource.data,
    save: async () => {
      const result = await emotionIllustrationRequest("illustration/config", {
        background_mode: mode,
        background_world: world,
        background_bg_css: bgCss,
        background_fg_css: fgCss,
        background_preview_name: preview?.name || "",
        background_preview_url: preview?.url || "",
      });
      const config = result.config || {};
      // 입력 중에 더 최신 편집이 생길 수 있으므로 입력값은 되돌리지 않고
      // 서버가 저장한 값만 기준선으로 삼는다.
      setBaseline(formValue(
        String(config.background_bg_css || ""),
        String(config.background_fg_css || ""),
        config.background_preview_url
          ? {
              name: String(config.background_preview_name || ""),
              url: String(config.background_preview_url),
            }
          : null,
        config.background_mode === "auto" ? "auto" : "manual",
        String(config.background_world || ""),
      ));
    },
  });

  const selected = presets.find(item => item.name === selectedPreset) || null;
  const selectedLocked = selected?.scope === "builtin" && !unlockToken;
  const dialogName = presetName.trim();
  const duplicateName = presets.some(item => item.name === dialogName
    && !(presetDialog === "rename" && item.name === selectedPreset));

  function selectPreset(name: string) {
    setSelectedPreset(name);
    const target = presets.find(item => item.name === name);
    if (!target) return;
    setBgCss(target.bg_css || "");
    setFgCss(target.fg_css || "");
    setPreview(target.preview_url ? { name: target.preview_name || "", url: target.preview_url } : null);
    setPresetError("");
    setPresetMessage(`'${target.name}' 프리셋을 불러왔습니다. 변경 사항은 자동 저장됩니다.`);
  }

  function openPresetDialog(type: PresetDialog) {
    setPresetError("");
    setPresetMessage("");
    if (type === "save") {
      setPresetName("");
      setPresetScope("local");
    } else if (type === "duplicate" && selected) {
      let next = `${selected.name} (복사)`;
      let index = 2;
      while (presets.some(item => item.name === next)) next = `${selected.name} (복사 ${index++})`;
      setPresetName(next);
      // 잠긴 배포 프리셋을 복제하면 사용자 프리셋으로 만드는 것이 안전하다.
      setPresetScope(selected.scope === "builtin" && !unlockToken ? "local" : selected.scope);
    } else if (type === "rename" && selected) {
      setPresetName(selected.name);
    }
    setPresetDialog(type);
  }

  async function mutatePreset(action: "save" | "duplicate" | "rename" | "delete" | "scope",
    payload: IllustrationData = {}) {
    if (presetBusy) return false;
    setPresetBusy(true);
    setPresetError("");
    setPresetMessage("");
    try {
      const result = await emotionIllustrationRequest("illustration/background-presets", {
        action, builtin_unlock_token: unlockToken, ...payload,
      });
      const next: BackgroundPreset[] = Array.isArray(result.presets) ? result.presets : [];
      setPresets(next);
      setPresetDialog(null);
      if (action === "save") {
        setSelectedPreset(String(payload.name));
        setPresetMessage(`'${payload.name}' 프리셋을 저장했습니다.`);
      } else if (action === "duplicate") {
        setSelectedPreset(String(payload.new_name));
        setPresetMessage(`'${payload.name}' 프리셋을 '${payload.new_name}'(으)로 복제했습니다.`);
      } else if (action === "rename") {
        setSelectedPreset(String(payload.new_name));
        setPresetMessage(`'${payload.name}' → '${payload.new_name}' 이름을 변경했습니다.`);
      } else if (action === "delete") {
        setSelectedPreset(current => current === payload.name ? "" : current);
        setPresetMessage(`'${payload.name}' 프리셋을 삭제했습니다.`);
      } else {
        setPresetMessage("프리셋 구분을 변경했습니다.");
      }
      return true;
    } catch (cause) {
      setPresetDialog(null);
      setPresetError(cause instanceof Error ? cause.message : "프리셋을 처리하지 못했습니다.");
      return false;
    } finally {
      setPresetBusy(false);
    }
  }

  async function unlockPresets() {
    if (presetBusy) return;
    setPresetBusy(true);
    setPresetError("");
    setPresetMessage("");
    try {
      const result = await emotionIllustrationRequest("illustration/background-presets", {
        action: "unlock", confirmed: true,
      });
      setUnlockToken(String(result.builtin_unlock_token || ""));
      if (Array.isArray(result.presets)) setPresets(result.presets);
      setPresetDialog(null);
      setPresetMessage("배포 잠금을 해제했습니다.");
    } catch (cause) {
      setPresetDialog(null);
      setPresetError(cause instanceof Error ? cause.message : "배포 잠금을 해제하지 못했습니다.");
    } finally {
      setPresetBusy(false);
    }
  }

  async function lockPresets() {
    if (presetBusy || !unlockToken) return;
    setPresetBusy(true);
    setPresetError("");
    setPresetMessage("");
    try {
      const result = await emotionIllustrationRequest("illustration/background-presets", {
        action: "lock", builtin_unlock_token: unlockToken,
      });
      setUnlockToken("");
      if (Array.isArray(result.presets)) setPresets(result.presets);
      setPresetMessage("배포 잠금을 설정했습니다.");
    } catch (cause) {
      setPresetError(cause instanceof Error ? cause.message : "배포 잠금을 설정하지 못했습니다.");
    } finally {
      setPresetBusy(false);
    }
  }

  function submitPresetDialog(event: FormEvent) {
    event.preventDefault();
    if (presetBusy) return;
    if (presetDialog === "unlock") {
      void unlockPresets();
      return;
    }
    if (presetDialog === "delete") {
      if (selected) void mutatePreset("delete", { name: selected.name });
      return;
    }
    if (!dialogName || duplicateName) return;
    if (presetDialog === "save") {
      void mutatePreset("save", {
        name: dialogName, scope: presetScope,
        bg_css: bgCss, fg_css: fgCss,
        preview_name: preview?.name || "", preview_url: preview?.url || "",
      });
    } else if (presetDialog === "duplicate") {
      if (selected) void mutatePreset("duplicate", {
        name: selected.name, new_name: dialogName, scope: presetScope,
      });
    } else if (presetDialog === "rename") {
      if (selected) void mutatePreset("rename", {
        name: selected.name, new_name: dialogName,
      });
    }
  }

  if (!resource.data) return <Panel title="배경 설정">
    <IllustrationLoading error={resource.error} retry={resource.reload} />
  </Panel>;

  return <div className="illust-stack">
    <Panel title="배경 설정">
      <div className="illust-body illust-form">
        <div className="illust-preset-picker">
          <label className="field">배경 모드
            <select value={mode} onChange={event => onModeChange(event.target.value as BackgroundMode)}>
              {(Object.keys(backgroundModes) as BackgroundMode[]).map(key =>
                <option key={key} value={key}>{backgroundModes[key]}</option>)}
            </select>
          </label>
        </div>
      </div>
    </Panel>

    <div hidden={mode !== "manual"}>
      <Panel title={backgroundModes.manual} aside={<span role="status">{autosave.status}</span>}>
        <div className="illust-body illust-form">
          <p className="field-help">{modeDescriptions.manual}</p>
          <form ref={form} className="illust-form" onSubmit={event => { event.preventDefault(); void autosave.retry(); }}>
            <div className="illust-manual-layout">
              <section className="illust-manual-preview-column" aria-label="수동 배경 미리보기">
                <div className="illust-actions">
                  <button type="button" className="button" onClick={() => setPicker(true)}>
                    <Images size={14} />백업에서 이미지 선택
                  </button>
                  {preview && <button type="button" className="button" onClick={() => setPreview(null)}>선택 해제</button>}
                </div>
                <ManualBackgroundPreview imageUrl={previewUrl} bgCss={bgCss} fgCss={fgCss} ratio={ratio} />
                <p className="field-help illust-manual-preview-name">
                  {preview?.name || "이미지를 선택하지 않아도 CSS만으로 미리보기를 확인할 수 있습니다."}
                </p>
              </section>

              <div className="illust-manual-side">
                <section className="illust-manual-presets" aria-label="배경 프리셋">
                  <header className="illust-manual-group-head">
                    <h3>배경 프리셋</h3>
                    {selected && <Badge>{selected.scope === "builtin"
                      ? (unlockToken ? "배포 · 편집 가능" : "배포 · 읽기 전용")
                      : "사용자 프리셋"}</Badge>}
                  </header>
                  <label className="field">저장된 프리셋
                    <SearchableSelect aria-label="배경 프리셋" value={selectedPreset}
                      disabled={presetBusy || !presetsResource.data}
                      onChange={event => selectPreset(event.target.value)}>
                      <option value="">{!presetsResource.data ? "프리셋을 불러오는 중…"
                        : presets.length ? "프리셋 선택" : "저장된 프리셋 없음"}</option>
                      {(["builtin", "local"] as const).map(scope => {
                        const group = presets.filter(item => item.scope === scope);
                        return group.length ? <optgroup key={scope} label={presetScopeLabels[scope]}>
                          {group.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}
                        </optgroup> : null;
                      })}
                    </SearchableSelect>
                  </label>
                  <div className="illust-actions">
                    <button type="button" className="button primary" disabled={presetBusy}
                      onClick={() => openPresetDialog("save")}><Save size={14} />현재 설정 저장</button>
                    <button type="button" className="button" disabled={presetBusy || !selected}
                      onClick={() => openPresetDialog("duplicate")}><Copy size={14} />프리셋 복제</button>
                    <button type="button" className="button" disabled={presetBusy || !selected || selectedLocked}
                      onClick={() => openPresetDialog("rename")}><Pencil size={14} />이름 변경</button>
                    <button type="button" className="button danger" disabled={presetBusy || !selected || selectedLocked}
                      onClick={() => openPresetDialog("delete")}><Trash2 size={14} />삭제</button>
                    <button type="button" className="button" disabled={presetBusy}
                      onClick={() => unlockToken ? void lockPresets() : openPresetDialog("unlock")}>
                      {unlockToken ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}
                      {unlockToken ? "배포 잠그기" : "배포 풀기"}
                    </button>
                  </div>
                  {selected && <label className="field">프리셋 구분
                    <select value={selected.scope} disabled={presetBusy || selectedLocked}
                      onChange={event => void mutatePreset("scope", {
                        name: selected.name,
                        scope: event.target.value as PresetScope,
                      })}>
                      <option value="local">사용자 프리셋</option>
                      <option value="builtin" disabled={!unlockToken}>배포 프리셋</option>
                    </select>
                    <span className="field-help">{selectedLocked
                      ? "배포 프리셋은 읽기 전용입니다. 복제하거나 배포 풀기를 눌러 편집하세요."
                      : selected.scope === "builtin"
                        ? "앱 업데이트 시 배포 파일 전체가 백업 없이 교체되어 변경 내용이 덮어씌워집니다."
                        : "사용자 프리셋은 업데이트 후에도 유지됩니다."}</span>
                  </label>}
                  <IllustrationFeedback error={presetError || presetsResource.error} message={presetMessage} />
                  {!presets.length && <p className="field-help">
                    {presetsResource.data ? "저장된 프리셋이 없습니다." : "프리셋을 불러오는 중…"}
                  </p>}
                </section>

                <section className="illust-manual-css-group" aria-label="배경·전면 CSS">
                  <header className="illust-manual-group-head"><h3>레이어 CSS</h3></header>
                  <p className="field-help">아래 → 위 순서: 배경 CSS · 불러온 이미지 · 전면 CSS</p>
                  <label className="field">배경 CSS
                    <textarea rows={8} className="illust-code" value={bgCss} maxLength={20000} spellCheck={false}
                      placeholder={".background { background: radial-gradient(circle at 50% 30%, #ffd7e2, #51415e); }"}
                      onChange={event => setBgCss(event.target.value)}
                      onBlur={() => void autosave.retry()} />
                  </label>
                  <p className="field-help">
                    맨 아래 레이어입니다. 선택자 없이 선언만 입력하면 <code>.background</code>에 적용됩니다.
                    불러온 이미지는 그 위 <code>.image</code>, 전면 CSS는 맨 위 <code>.foreground</code> 레이어입니다.
                  </p>
                  <label className="field">전면 CSS
                    <textarea rows={8} className="illust-code" value={fgCss} maxLength={20000} spellCheck={false}
                      placeholder={".foreground { background: linear-gradient(180deg, transparent 55%, #0008); }"}
                      onChange={event => setFgCss(event.target.value)}
                      onBlur={() => void autosave.retry()} />
                  </label>
                  <p className="field-help">
                    배경 이미지 위에 겹치는 빈 레이어입니다. 그라데이션·비네트·프레임 등 전면 효과를 CSS로 넣으세요.
                  </p>
                </section>
              </div>
            </div>
            <p className="field-help">변경 사항은 자동 저장됩니다. 입력을 멈추거나 입력칸을 벗어나면 저장됩니다.</p>
            {(autosave.error || autosave.inputError) && <div className="notice error" role="alert">
              {autosave.error || autosave.inputError}
              {autosave.error && <button type="button" className="button" onClick={() => void autosave.retry()}>저장 다시 시도</button>}
            </div>}
          </form>
        </div>
      </Panel>
    </div>

    <div hidden={mode !== "auto"}>
      {mode === "auto" && <EmotionAutoBackgroundPresets onPresetChange={onPresetChange}
        toolbox={<Panel title="툴박스" aside={<span role="status">{autosave.status}</span>}>
          <div className="illust-body illust-form">
            <div className="illust-bot-toolbox">
              <button type="button" className="button secondary" aria-haspopup="dialog" aria-expanded={worldOpen}
                onClick={() => setWorldOpen(true)}><Globe size={14} />세계관 설정</button>
              <button type="button" className="button" aria-haspopup="dialog" aria-expanded={!!generationBot}
                disabled={!botSelected} onClick={() => setGenerationBot(botSelected)}><SlidersHorizontal size={14} />그리기 설정</button>
              <button type="button" className="button" aria-haspopup="dialog" aria-expanded={compositionOpen}
                onClick={() => setCompositionOpen(true)}><Blend size={14} />합성 설정</button>
            </div>
            <p className="field-help">{botSelected ? `현재 감정 삽화 봇 '${botSelected}'의 그리기 설정을 함께 사용합니다.` : "감정 삽화 봇 설정에서 봇을 먼저 선택하세요."}</p>
            <IllustrationFeedback error={autosave.error || autosave.inputError} />
          </div>
        </Panel>} />}
    </div>

    {compositionOpen && <Modal title="합성 설정" className="emotion-composition-modal" backgroundable={false} closeDisabled={compositionBusy}
      onClose={() => { if (compositionDirty) setDiscardComposition(true); else setCompositionOpen(false); }}>
      <EmotionCompositionSettings botSelected={botSelected} onDirtyChange={setCompositionDirty} onBusyChange={setCompositionBusy} />
    </Modal>}
    {discardComposition && <Modal title="설정 닫기" backgroundable={false} closeDisabled={compositionBusy} onClose={() => setDiscardComposition(false)}>
      <p>저장하지 않은 변경을 버리고 닫을까요?</p>
      <div className="modal-actions">
        <button type="button" className="button" onClick={() => setDiscardComposition(false)}>계속 편집</button>
        <button type="button" className="button danger" onClick={() => { setDiscardComposition(false); setCompositionDirty(false); setCompositionOpen(false); }}>변경 버리기</button>
      </div>
    </Modal>}

    {generationBot && <Modal title={`${generationBot} · 그리기 설정`} backgroundable={false}
      className="illust-character-modal" closeDisabled={generationBusy}
      onClose={() => { if (generationDirty) setDiscardGeneration(true); else setGenerationBot(""); }}>
      <IllustrationFeedback error={distribution.error} />
      <EmotionBotApiProvider><IllustBotGenerationSettings key={generationBot} botName={generationBot} initialMode="background"
        readOnly={!distribution.data?.bots?.some((bot: IllustrationData) => bot.name === generationBot && bot.scope === "local")}
        onDirtyChange={setGenerationDirty} onBusyChange={setGenerationBusy} /></EmotionBotApiProvider>
    </Modal>}
    {discardGeneration && <Modal title="설정 닫기" backgroundable={false} onClose={() => setDiscardGeneration(false)}>
      <p>저장하지 않은 변경을 버리고 닫을까요?</p>
      <div className="modal-actions"><button type="button" className="button" onClick={() => setDiscardGeneration(false)}>계속 편집</button>
        <button type="button" className="button danger" onClick={() => { setDiscardGeneration(false); setGenerationDirty(false); setGenerationBot(""); }}>변경 버리기</button></div>
    </Modal>}

    {worldOpen && <Modal title="세계관 설정" backgroundable={false} onClose={() => setWorldOpen(false)}>
      <div className="illust-form">
        <label className="field">세계관 설정
          <textarea rows={12} value={world} maxLength={20000}
            placeholder="배경의 시대, 건축 양식, 지역, 기술 수준 등을 입력하세요."
            onChange={event => setWorld(event.target.value)} onBlur={() => void autosave.retry()} />
          <span className="field-help">자동 배경 모드에서 원문과 함께 배경 선택·생성에 사용합니다.</span>
        </label>
        <span role="status">{autosave.status}</span>
        <IllustrationFeedback error={autosave.error || autosave.inputError} />
        <div className="modal-actions">
          {autosave.error && <button type="button" className="button" onClick={() => void autosave.retry()}>저장 다시 시도</button>}
          <button type="button" className="button" onClick={() => setWorldOpen(false)}>닫기</button>
        </div>
      </div>
    </Modal>}

    {picker && <ManualBackgroundPicker onClose={() => setPicker(false)} onSelect={item => {
      setPreview({ name: item.name, url: item.raw_url || item.image_url });
      setPicker(false);
    }} />}

    {presetDialog && <Modal
      title={presetDialog === "save" ? "배경 프리셋 저장"
        : presetDialog === "duplicate" ? "배경 프리셋 복제"
        : presetDialog === "rename" ? "배경 프리셋 이름 변경"
        : presetDialog === "unlock" ? "배포 풀기"
        : "배경 프리셋 삭제"}
      backgroundable={false} closeDisabled={presetBusy} onClose={() => setPresetDialog(null)}>
      <form className="illust-form" onSubmit={submitPresetDialog}>
        {presetDialog === "delete"
          ? <p>'{selected?.name}' 프리셋을 삭제할까요? 현재 화면의 설정은 유지됩니다.</p>
          : presetDialog === "unlock"
            ? <>
              <p>배포 프리셋을 직접 편집할 수 있도록 잠금을 해제합니다.</p>
              <p><strong>업데이트 시 배포 파일 전체가 백업 없이 교체되어 수정한 내용이 덮어씌워집니다.</strong></p>
              <p className="field-help">설정을 계속 보관하려면 프리셋 복제로 사용자 프리셋을 만들어 사용하세요.</p>
            </>
          : <>
            <label className="field">프리셋 이름
              <input autoFocus required maxLength={100} value={presetName} disabled={presetBusy}
                onChange={event => setPresetName(event.target.value)} />
              {duplicateName && <span className="field-error">같은 이름의 프리셋이 있습니다.</span>}
            </label>
            {presetDialog !== "rename" && <label className="field">프리셋 구분
              <select value={presetScope} disabled={presetBusy}
                onChange={event => setPresetScope(event.target.value as PresetScope)}>
                <option value="local">사용자 프리셋</option>
                <option value="builtin" disabled={!unlockToken}>배포 프리셋</option>
              </select>
              <span className="field-help">{presetScope === "builtin"
                ? "앱 업데이트 시 배포 파일 전체가 백업 없이 교체되어 변경 내용이 덮어씌워집니다."
                : unlockToken ? "사용자 프리셋은 업데이트 후에도 유지됩니다."
                : "사용자 프리셋은 업데이트 후에도 유지됩니다. 배포 프리셋으로 저장하려면 배포 풀기가 필요합니다."}</span>
            </label>}
            {presetDialog === "duplicate" && <p className="field-help">
              '{selected?.name}' 프리셋의 CSS·미리보기 이미지를 복사합니다.
            </p>}
          </>}
        <IllustrationFeedback error={presetError} />
        <div className="illust-actions">
          <button type="button" className="button" disabled={presetBusy}
            onClick={() => setPresetDialog(null)}>취소</button>
          <button type="submit" className={`button ${presetDialog === "delete" ? "danger" : "primary"}`}
            disabled={presetBusy || (presetDialog !== "delete" && presetDialog !== "unlock" && (!dialogName || duplicateName))}>
            {presetBusy ? "처리 중…"
              : presetDialog === "save" ? "저장"
              : presetDialog === "duplicate" ? "복제"
              : presetDialog === "rename" ? "이름 변경"
              : presetDialog === "unlock" ? "배포 풀기" : "삭제"}
          </button>
        </div>
      </form>
    </Modal>}
  </div>;
}

/**
 * 레이어 순서(아래 → 위): 배경 CSS(.background) → 불러온 이미지(.image)
 * → 전면 CSS(.foreground). 사용자가 입력한 CSS가 공방 UI 전체로 새어
 * 나가지 않도록 Shadow DOM 안에서 렌더링한다.
 */
function ManualBackgroundPreview({ imageUrl, bgCss, fgCss, ratio }: {
  imageUrl: string; bgCss: string; fgCss: string; ratio: number;
}) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = host.current;
    if (!node) return;
    const root = node.shadowRoot || node.attachShadow({ mode: "open" });
    const imageValue = imageUrl
      ? `url("${imageUrl.replace(/["\\\n\r]/g, "\\$&")}")`
      : "none";
    const style = document.createElement("style");
    style.textContent = [
      `:host{--illust-preview-image:${imageValue}}`,
      ".stage{position:relative;width:100%;height:100%;overflow:hidden;background:#17131c}",
      ".background{position:absolute;inset:0}",
      ".image{position:absolute;inset:0;background-image:var(--illust-preview-image,none);background-position:center;background-size:cover;background-repeat:no-repeat}",
      ".foreground{position:absolute;inset:0;pointer-events:none}",
      layerRules(".background", bgCss),
      layerRules(".foreground", fgCss),
    ].filter(Boolean).join("\n");
    const stage = document.createElement("div");
    stage.className = "stage";
    const background = document.createElement("div");
    background.className = "background";
    const image = document.createElement("div");
    image.className = "image";
    const foreground = document.createElement("div");
    foreground.className = "foreground";
    stage.append(background, image, foreground);
    root.replaceChildren(style, stage);
  }, [imageUrl, bgCss, fgCss]);

  return <div ref={host} className="illust-manual-preview" style={{ aspectRatio: ratio }}
    aria-label="수동 배경 미리보기" />;
}

function ManualBackgroundPicker({ onClose, onSelect }: {
  onClose: () => void; onSelect: (item: IllustrationData) => void;
}) {
  const resource = useEmotionIllustrationResource("backups?filter=all&limit=24&offset=0");
  const items: IllustrationData[] = Array.isArray(resource.data?.backups) ? resource.data.backups : [];
  return <Modal title="수동 배경 미리보기 이미지" backgroundable={false} onClose={onClose}>
    <p className="field-help">최근 감정 삽화 백업에서 미리보기 배경으로 사용할 이미지를 선택합니다.</p>
    <IllustrationFeedback error={resource.error} />
    {!resource.data
      ? <IllustrationLoading error={resource.error} retry={resource.reload} />
      : !items.length
        ? <Empty title="백업 이미지가 없습니다">감정 삽화 백업을 먼저 생성한 뒤 다시 선택하세요.</Empty>
        : <div className="illust-postprocess-backups">{items.map(item => <button type="button"
            className="illust-reference-preview" key={item.name} title={item.name} aria-label={item.name}
            onClick={() => onSelect(item)}>
            <img loading="lazy" src={illustrationMedia(item.raw_url || item.image_url)} alt={item.name} />
          </button>)}</div>}
  </Modal>;
}
