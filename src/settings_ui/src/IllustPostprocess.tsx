import { useEffect, useState } from "react";
import { Save } from "lucide-react";
import { Modal, Panel, SettingsTabs } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { illustrationRequest, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";
import { IllustPostprocessPreview } from "./IllustPostprocessPreview";

const labels: Record<string, string> = {
  enabled: "대사 합성 사용", placement: "배치", height_mode: "대사 영역 높이 단위", height_value: "대사 영역 높이", font_size: "글자 크기 (px)", name_font_size: "이름 글자 크기 (0: 자동)", emotion_font_size: "감정 글자 크기 (0: 자동)",
  name_color: "이름에 캐릭터 색상 적용", dialogue_color: "대사에 캐릭터 색상 적용", text_outline_width: "글자 외곽 두께 (-1: 자동)", name_replace_enabled: "이름 치환 사용", strip_emotion: "대사에서 감정 태그 숨기기", prefix: "이미지 조회 접두사", suffix: "이미지 조회 접미사", face_enabled: "얼굴 표시", multi_face_mode: "여러 화자의 얼굴 표시", theme_single: "1인 테마", theme_dual: "2인 이상 테마", opacity: "배경 불투명도",
  font_id: "폰트", min_font_size: "최소 글자 크기 (0: 자동)", letter_spacing: "자간", line_height_ratio: "행간 배율", text_width_scale: "글자 가로 배율", layout_font_scale: "글자 확대 상한 배율", text_color: "글자 색상", bubble_fill: "말풍선 배경 색상", bubble_border: "말풍선 테두리 색상", border_width: "테두리 두께", svg_border_width: "효과 말풍선 테두리 두께", speech_opacity: "대사 말풍선 불투명도", thought_opacity: "생각 말풍선 불투명도", padding: "안쪽 여백", radius: "모서리 크기", thought_shape: "생각 말풍선 모양", tail_threshold: "꼬리 생성 최대 거리 배율", bubble_shape: "말풍선 외곽선", tail_width_scale: "꼬리 두께 배율", tail_max_length: "꼬리 최대 길이 (px)", organic_wobble: "외곽선 굴곡 강도", max_width_ratio: "이미지 대비 최대 너비 비율", onnx_device: "얼굴 탐지 장치", cpu_threads: "CPU 스레드 (0: 자동)", speech_split: "긴 대사 말풍선 분할",
  bottom_margin_ratio: "아래 여백 비율", line_spacing_ratio: "자막 줄 간격 비율", outline_color: "외곽선 색상", outline_width: "외곽선 두께", shadow_color: "그림자 색상", shadow_opacity: "그림자 불투명도", shadow_offset_x: "그림자 가로 위치", shadow_offset_y: "그림자 세로 위치", thought_italic_enabled: "생각 대사 기울임", thought_italic_shear: "기울임 정도", max_lines: "최대 줄 수",
  match_thres: "얼굴 일치 기준", face_candidates_per_character: "캐릭터별 얼굴 후보 수", appearance_weight: "외모 비교 가중치", assignment_ambiguity_margin: "얼굴 배정 모호성 기준", face_fallback: "얼굴 탐지 실패 시 대체 배치",
};
const themes: [string, string][] = [["sky", "하늘색 / 은색"], ["ivory", "아이보리 / 금색"], ["lavender", "연보라 / 핑크"], ["black", "검정 / 은회색"], ["gray", "회색 / 밝은 글자"], ["classic", "검정 심플"], ["gfl", "전술 HUD · 청록 / 주황"]];
const postprocessGroups = [
  { title: "합성 사용·배치", keys: ["enabled", "placement", "height_mode", "height_value", "max_width_ratio", "bottom_margin_ratio", "max_lines"] },
  { title: "글자·줄 구성", keys: ["font_id", "font_size", "min_font_size", "name_font_size", "emotion_font_size", "letter_spacing", "line_height_ratio", "text_width_scale", "layout_font_scale", "line_spacing_ratio"] },
  { title: "색상·외곽선", keys: ["text_color", "name_color", "dialogue_color", "text_outline_width", "outline_color", "outline_width", "shadow_color", "shadow_opacity", "shadow_offset_x", "shadow_offset_y", "bubble_fill", "bubble_border", "border_width", "svg_border_width", "speech_opacity", "thought_opacity"] },
  { title: "말풍선 모양·꼬리", keys: ["thought_shape", "bubble_shape", "radius", "padding", "tail_threshold", "tail_width_scale", "tail_max_length", "organic_wobble", "speech_split"] },
  { title: "대사창 테마·얼굴", keys: ["theme_single", "theme_dual", "opacity", "face_enabled", "multi_face_mode"] },
  { title: "얼굴 탐지", keys: ["match_thres", "face_candidates_per_character", "appearance_weight", "assignment_ambiguity_margin", "face_fallback", "onnx_device", "cpu_threads"] },
  { title: "이름·감정 표시", keys: ["strip_emotion", "name_replace_enabled", "prefix", "suffix", "thought_italic_enabled", "thought_italic_shear"] },
];
const options: Record<string, [string, string][]> = { placement: [["overlay", "이미지 위에 배치"], ["extend", "이미지 아래로 확장"]], height_mode: [["ratio", "비율"], ["px", "픽셀"]], multi_face_mode: [["both", "앞의 두 화자"], ["first", "첫 화자만"]], thought_shape: [["cloud", "구름"], ["box", "상자"]], bubble_shape: [["legacy", "기본"], ["organic", "굴곡형"]], theme_single: [...themes, ["nikke", "사선 프레임 · 검정 / 청록"], ["devil", "하단 그라데이션 · 검정 / 주황"]], theme_dual: [["classic_simple", "심플 배치 · 검정 심플"], ["gfl_simple", "심플 배치 · 전술 HUD"], ["devil_simple", "심플 배치 · 하단 그라데이션"], ["nikke_simple", "심플 배치 · 사선 프레임"], ...themes.map(([id, label]): [string, string] => [id, `블럭형 배치 · ${label}`])] };

export function IllustPostprocess({ botName, active }: { botName: string; active: boolean }) {
  const [data, setData] = useState<IllustrationData | null>(null); const [baseline, setBaseline] = useState<IllustrationData | null>(null); const [error, setError] = useState(""); const [revision, setRevision] = useState(0);
  const [mode, setMode] = useState("vn"); const [emotionRules, setEmotionRules] = useState("");
  const action = useIllustrationAction();
  const fonts = useIllustrationResource("postprocess/fonts"); const devices = useIllustrationResource("postprocess/face_devices");
  useEffect(() => {
    const controller = new AbortController(); setError("");
    void illustrationRequest("postprocess/settings", undefined, controller.signal).then(value => {
      if (controller.signal.aborted) return;
      setData(value); setBaseline(value); setMode(value.mode); setEmotionRules(JSON.stringify(value.vn.emotion_extract_rules || [], null, 2));
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [revision]);
  function update(key: string, value: unknown) { setData(current => current && ({ ...current, [mode]: { ...current[mode], [key]: value } })); }
  function settings(): IllustrationData { const rules = JSON.parse(emotionRules); if (!Array.isArray(rules)) throw new Error("감정 추출 규칙은 배열이어야 합니다."); return { ...data, vn: { ...data!.vn, emotion_extract_rules: rules } }; }
  const invalidNumber = data && Object.keys(baseline?.[mode] || {}).find(key => typeof baseline![mode][key] === "number" && (typeof data[mode][key] !== "number" || !Number.isFinite(data[mode][key])));
  return <Panel title="대사 합성 설정" className="illust-postprocess-panel"><div className="illust-body">
    {!data ? <IllustrationLoading error={error} retry={() => setRevision(value => value + 1)} /> : <form className="illust-form" onSubmit={event => { event.preventDefault(); void action.run(async () => {
      const values = settings();
      const changed = Object.fromEntries(["vn", "bubble", "subtitle"].map(type => [type, Object.fromEntries(Object.entries(values[type]).filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(baseline![type][key])))]));
      const saved = await illustrationRequest("postprocess/settings", { ...changed, ...(mode !== baseline!.mode ? { mode } : {}) });
      setData(saved); setBaseline(saved); setMode(saved.mode); setEmotionRules(JSON.stringify(saved.vn.emotion_extract_rules || [], null, 2));
    }); }}><fieldset className="illust-form" disabled={action.busy}>
      <SettingsTabs id="illust-postprocess-mode" label="대사 합성 방식" values={{ vn: "대사창", bubble: "말풍선", subtitle: "자막" }} value={mode} onChange={setMode} />
      <div className="illust-postprocess-workspace"><div className="illust-form">
      <div role="tabpanel" id={`illust-postprocess-mode-panel-${mode}`} aria-labelledby={`illust-postprocess-mode-tab-${mode}`} className="illust-form">
      {postprocessGroups.filter(group => group.keys.some(key => key in data[mode])).map(group => <fieldset className="illust-postprocess-group" key={group.title}><legend>{group.title}</legend>
      <div className="illust-fields">{group.keys.filter(key => key in data[mode]).map(key => { const value = data[mode][key];
        const list = key === "font_id" ? fonts.data?.fonts?.map((font: IllustrationData) => [font.id, font.name]) : key === "onnx_device" ? devices.data?.devices?.map((device: IllustrationData) => [device.key, device.label]) : options[key];
        return typeof value === "boolean" ? <label className="illust-check" key={key}><input type="checkbox" checked={value} onChange={event => update(key, event.target.checked)} />{labels[key]}</label> : <label className="field" key={key}>{labels[key]}{list ? <select value={String(value)} onChange={event => update(key, event.target.value)}>{!list.some(([id]: string[]) => id === value) && <option value={String(value)}>{String(value)}</option>}{list.map(([id, text]: string[]) => <option value={id} key={id}>{text}</option>)}</select> : <input required={typeof baseline![mode][key] === "number"} type={typeof baseline![mode][key] === "number" ? "number" : typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value) ? "color" : "text"} step="any" value={String(value ?? "")} onChange={event => update(key, typeof baseline![mode][key] === "number" && event.target.value !== "" ? Number(event.target.value) : event.target.value)} />}</label>;
      })}</div></fieldset>)}
      </div>
      {mode !== "vn" && <FontManager resource={fonts} />}
      <IllustrationFeedback error={devices.error} />
      {mode === "vn" && <details className="illust-section"><summary>이름 치환·감정 추출</summary><div className="illust-form">{Object.entries(data.vn.name_replace || {}).map(([name, value], index) => <div className="illust-fields" key={index}><label className="field">원래 이름<input value={name} onChange={event => update("name_replace", Object.fromEntries(Object.entries(data.vn.name_replace).map(([key, text]) => [key === name ? event.target.value : key, text])))} /></label><label className="field">표시할 이름<input value={String(value)} onChange={event => update("name_replace", { ...data.vn.name_replace, [name]: event.target.value })} /></label><button type="button" className="button danger" onClick={() => update("name_replace", Object.fromEntries(Object.entries(data.vn.name_replace).filter(([key]) => key !== name)))}>이름 치환 삭제</button></div>)}<button type="button" className="button" onClick={() => update("name_replace", { ...data.vn.name_replace, "": "" })}>이름 치환 추가</button><label className="field">감정 추출 규칙 (JSON)<textarea rows={5} value={emotionRules} onChange={event => setEmotionRules(event.target.value)} /></label></div></details>}
      <IllustrationFeedback {...action} /><button className="button primary"><Save size={14} />합성 설정 저장</button>
      </div><IllustPostprocessPreview mode={mode} settings={data[mode]} emotionRules={emotionRules} validation={invalidNumber ? `${labels[invalidNumber] || invalidNumber} 값을 입력하세요.` : ""} botName={botName} active={active} /></div>
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
