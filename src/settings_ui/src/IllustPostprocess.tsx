import { useEffect, useState } from "react";
import { Save } from "lucide-react";
import { Modal, Panel, Toggle } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { illustrationRequest, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";
import { IllustPostprocessPreview } from "./IllustPostprocessPreview";

const labels: Record<string, string> = {
  enabled: "대사 합성 사용", font_size: "글자 크기 (px)", font_weight: "글자 굵기",
  font_id: "폰트", min_font_size: "최소 글자 크기 (px)", letter_spacing: "자간", line_height_ratio: "행간 배율", text_width_scale: "글자 가로 배율", text_color: "글자 색상", max_width_ratio: "이미지 대비 최대 너비 비율",
  bottom_margin_ratio: "아래 여백 비율", line_spacing_ratio: "자막 줄 간격 비율", outline_color: "외곽선 색상", outline_width: "외곽선 두께", shadow_color: "그림자 색상", shadow_opacity: "그림자 불투명도", shadow_offset_x: "그림자 가로 위치", shadow_offset_y: "그림자 세로 위치", thought_italic_enabled: "생각 대사 기울임", thought_italic_shear: "기울임 정도", max_lines: "최대 줄 수",
  erase_mode: "지우기 방식", erase_padding: "지우기 여백 (px)", write_area: "재작성 위치", match_max_distance: "JSON 항목 매칭 최대 거리 (px)", text_align: "글자 정렬", text_padding: "글자 안쪽 여백 (px)", detect_cpu_threads: "탐지 CPU 스레드 (0: 자동)",
};
const postprocessModes = { subtitle: "자막", manga_text: "텍스트 교체" };
const postprocessGroups = [
  { title: "합성 사용·배치", keys: ["enabled", "max_width_ratio", "bottom_margin_ratio", "max_lines"] },
  { title: "글자·줄 구성", keys: ["font_id", "font_size", "min_font_size", "font_weight", "letter_spacing", "line_height_ratio", "text_width_scale", "line_spacing_ratio"] },
  { title: "색상·외곽선", keys: ["text_color", "outline_color", "outline_width", "shadow_color", "shadow_opacity", "shadow_offset_x", "shadow_offset_y"] },
  { title: "감정 표시", keys: ["thought_italic_enabled", "thought_italic_shear"] },
  { title: "텍스트 교체·지우기", keys: ["write_area", "erase_mode", "erase_padding", "match_max_distance", "text_padding", "text_align", "detect_cpu_threads"] },
];
const options: Record<string, [string, string][]> = { font_weight: [["300", "Light 300"], ["400", "Regular 400"], ["500", "Medium 500"], ["600", "SemiBold 600"], ["700", "Bold 700"], ["800", "ExtraBold 800"], ["900", "Black 900"]], erase_mode: [["auto", "자동 (배경 단색이면 채우기)"], ["flat", "단색 채우기"], ["inpaint", "인페인트"]], write_area: [["detected", "검출된 실제 글자 영역"], ["json_bbox", "제출 JSON bbox"]], text_align: [["center", "가운데"], ["left", "왼쪽"], ["right", "오른쪽"]] };

export function IllustPostprocess({ botName, active, outputMode }: { botName: string; active: boolean; outputMode: string }) {
  const [data, setData] = useState<IllustrationData | null>(null); const [baseline, setBaseline] = useState<IllustrationData | null>(null); const [error, setError] = useState(""); const [revision, setRevision] = useState(0);
  const mode = ["manga", "manga_both"].includes(outputMode) ? "manga_text" : "subtitle";
  const action = useIllustrationAction();
  const fonts = useIllustrationResource("postprocess/fonts");
  useEffect(() => {
    const controller = new AbortController(); setError("");
    void illustrationRequest("postprocess/settings", undefined, controller.signal).then(value => {
      if (controller.signal.aborted) return;
      setData(value); setBaseline(value);
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [revision]);
  function update(key: string, value: unknown) { setData(current => {
    if (!current) return current;
    const next = { ...current[mode], [key]: value };
    if (key === "auto_font_size" && value) {
      for (const field of ["font_size", "min_font_size"]) {
        if (typeof next[field] !== "number" || !Number.isFinite(next[field])) next[field] = baseline?.[mode]?.[field];
      }
    }
    return { ...current, [mode]: next };
  }); }
  const modeData = data?.[mode] || {};
  const invalidNumber = data && Object.keys(baseline?.[mode] || {}).find(key => !(modeData.auto_font_size && ["font_size", "min_font_size"].includes(key)) && typeof (baseline?.[mode] || {})[key] === "number" && (typeof modeData[key] !== "number" || !Number.isFinite(modeData[key])));
  function renderField(key: string) {
    const value = modeData[key];
    const list = key === "font_id" ? fonts.data?.fonts?.map((font: IllustrationData) => [font.id, font.name]) : options[key];
    const label = key === "enabled" && mode === "manga_text" ? "텍스트 교체 사용" : labels[key];
    const numeric = typeof baseline?.[mode]?.[key] === "number";
    return typeof value === "boolean" ? <label className="illust-check" key={key}><input type="checkbox" checked={value} onChange={event => update(key, event.target.checked)} />{label}</label> : <label className="field" key={key}>{label}{list ? <select value={String(value)} onChange={event => update(key, numeric ? Number(event.target.value) : event.target.value)}>{!list.some(([id]: string[]) => id === value) && <option value={String(value)}>{String(value)}</option>}{list.map(([id, text]: string[]) => <option value={id} key={id}>{text}</option>)}</select> : <input required={numeric} type={numeric ? "number" : typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value) ? "color" : "text"} step="any" value={String(value ?? "")} onChange={event => update(key, numeric && event.target.value !== "" ? Number(event.target.value) : event.target.value)} />}</label>;
  }
  return <Panel title="대사 합성 설정" className="illust-postprocess-panel"><div className="illust-body">
    {!data ? <IllustrationLoading error={error} retry={() => setRevision(value => value + 1)} /> : <form className="illust-form" onSubmit={event => { event.preventDefault(); void action.run(async () => {
      const changed = Object.fromEntries(Object.keys(postprocessModes).filter(type => data[type]).map(type => [type, Object.fromEntries(Object.entries(data[type]).filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(baseline?.[type]?.[key])))]));
      const saved = await illustrationRequest("postprocess/settings", changed);
      setData(saved); setBaseline(saved);
    }); }}><fieldset className="illust-form" disabled={action.busy}>
      <div role="status"><strong>대사 합성 방식: {postprocessModes[mode]} · 자동 선택</strong><p className="field-help">출력 방식에 따라 일반 삽화에는 자막을, 만화 삽화에는 텍스트 교체를 적용합니다.</p></div>
      <div className="illust-postprocess-workspace"><div className="illust-form">
      <div className="illust-form" role="group" aria-label={`${postprocessModes[mode]} 설정`}>
      {postprocessGroups.filter(group => group.keys.some(key => key in modeData)).map(group => <fieldset className="illust-postprocess-group" key={group.title}><legend>{group.title}</legend>
      <div className="illust-fields">{group.keys.filter(key => key in modeData && key !== "min_font_size").map(key => key === "font_size" ? <fieldset className="illust-postprocess-font-size" key={key}><legend>글자 크기</legend>
        {"auto_font_size" in modeData && <><div className="illust-postprocess-auto-size"><span>자동 추론</span><Toggle label="자동 추론" checked={modeData.auto_font_size === true} onChange={value => update("auto_font_size", value)} /></div><p className="field-help">감지된 글자 영역과 넣을 글자 수에 맞춰 크기를 자동으로 정합니다.</p></>}
        {!modeData.auto_font_size && <div className="illust-fields">{renderField("font_size")}{"min_font_size" in modeData && renderField("min_font_size")}</div>}
      </fieldset> : renderField(key))}</div></fieldset>)}
      </div>
      <FontManager resource={fonts} />
      <IllustrationFeedback {...action} /><button className="button primary"><Save size={14} />합성 설정 저장</button>
      </div><IllustPostprocessPreview mode={mode} settings={modeData} validation={invalidNumber ? `${labels[invalidNumber] || invalidNumber} 값을 입력하세요.` : ""} botName={botName} active={active} /></div>
    </fieldset></form>}
  </div></Panel>;
}

function FontManager({ resource }: { resource: ReturnType<typeof useIllustrationResource> }) {
  const action = useIllustrationAction(); const [deleting, setDeleting] = useState<IllustrationData | null>(null);
  return <details className="illust-section"><summary>폰트 관리</summary><div className="illust-form">
    <label className="field">폰트 추가 (.ttf, .otf, .ttc)<input type="file" accept=".ttf,.otf,.ttc" disabled={action.busy} onChange={event => {
      const file = event.target.files?.[0]; event.target.value = ""; if (!file) return;
      void action.run(async () => { const body = new FormData(); body.append("font", file); await illustrationRequest("postprocess/font/upload", body); resource.reload(); }, "폰트를 추가했습니다.");
    }} /></label>
    <IllustrationFeedback {...action} error={action.error || resource.error} />
    {resource.data?.fonts?.filter((font: IllustrationData) => font.source === "upload").map((font: IllustrationData) => <div className="illust-actions" key={font.id}><span>{font.name}</span><button type="button" className="button danger" disabled={action.busy} onClick={() => setDeleting(font)}>폰트 삭제</button></div>)}
    {deleting && <Modal title="폰트 삭제" backgroundable={false} closeDisabled={action.busy} onClose={() => setDeleting(null)}><div className="illust-form"><p>{deleting.name} 폰트를 삭제할까요?</p><IllustrationFeedback error={action.error} /><div className="illust-actions"><button type="button" className="button danger" disabled={action.busy} onClick={() => void action.run(async () => { await illustrationRequest("postprocess/font/delete", { font_id: deleting.id }); resource.reload(); setDeleting(null); }, "폰트를 삭제했습니다.")}>삭제 확인</button><button type="button" className="button" disabled={action.busy} onClick={() => setDeleting(null)}>취소</button></div></div></Modal>}
  </div></details>;
}
