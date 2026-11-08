import type { FlowNode } from "./illustrationFlowLayout";

type ResultRecord = Record<string, unknown>;
type StateTable = {
  slots: (string | number)[];
  characters: string[];
  states?: Record<string, Record<string, ResultRecord>>;
  changes?: Record<string, Record<string, ResultRecord>>;
};
type StateKind = "profile" | "timePlace" | "outfit";

const stateViews = {
  profile: {
    title: "PROFILE-RESOLVE 프로필 시간표",
    label: "PROFILE-RESOLVE 구간별 프로필·판단 근거",
    help: "파란색은 확정된 시작·변경점이고, 회색은 앞선 프로필과 근거를 이어받은 값입니다. START는 첫 등장 시의 상태이며, 이후 C 칸은 해당 원문 구간이 끝난 시점의 상태입니다.",
    fields: [["profile", "적용 카드 ID"], ["reason", "판단 근거"]],
  },
  timePlace: {
    title: "TIME & PLACE 시간표",
    label: "TIME & PLACE 슬롯별 시간·장소·조명 상태",
    help: "왼쪽에서 오른쪽으로 상태를 이어받습니다. 파란색은 TIME & PLACE가 직접 출력한 값이고, 회색은 앞 SLOT에서 이어받은 값입니다. 각 항목은 새 값이 나오는 지점부터 독립적으로 바뀝니다.",
    fields: [["time", "TIME"], ["place", "PLACE"], ["lighting", "LIGHTING"]],
  },
  outfit: {
    title: "복장·헤어 시간표",
    label: "OUTFIT 슬롯별 복장·헤어 상태",
    help: "왼쪽에서 오른쪽으로 상태를 이어받습니다. 파란색은 OUTFIT가 직접 출력한 값이고, 회색은 앞 SLOT에서 이어받은 값입니다. 각 항목은 새 값이 나오는 지점부터 독립적으로 바뀝니다.",
    fields: [["outfit", "복장"], ["hair", "헤어"]],
  },
} as const;

function record(value: unknown): ResultRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as ResultRecord : null;
}
const resultText = (value: unknown, fallback = "—") => value == null ? fallback : typeof value === "string" ? value : JSON.stringify(value);
const characterName = (value: unknown) => typeof value === "string" ? value : resultText(record(value)?.name);
const slotKey = (slot: string | number) => typeof slot === "number" ? `SLOT${slot}` : slot;

function ResolveTable({ title, headers, rows, empty }: { title: string; headers: string[]; rows: (string | number)[][]; empty: string }) {
  return <section className="illust-flow-results">
    <h3>{title}</h3>
    <div className="illust-flow-table-scroll" role="region" aria-label={title} tabIndex={0}>
      <table className="illust-flow-result-table illust-flow-resolve-table" aria-label={title}>
        <thead><tr>{headers.map(header => <th scope="col" key={header}>{header}</th>)}</tr></thead>
        <tbody>{rows.map((cells, index) => <tr key={index}>{cells.map((value, column) => column === 0
          ? <th scope="row" key={column}>{value}</th> : <td key={column}>{value}</td>)}</tr>)}
          {!rows.length && <tr><td colSpan={headers.length}>{empty}</td></tr>}
        </tbody>
      </table>
    </div>
  </section>;
}

function CharacterResolveTables({ output }: { output: unknown }) {
  if (output == null || output === "") return null;
  let resolved = output;
  try {
    if (typeof output === "string") resolved = JSON.parse(output.slice(output.indexOf("{"), output.lastIndexOf("}") + 1));
  } catch {
    return <p className="field-help">표로 정리할 수 없는 응답입니다. 아래 출력을 확인하세요.</p>;
  }
  const data = record(resolved);
  if (!data) return null;
  const characters = Array.isArray(data.characters) ? data.characters : null;
  const scenes = Array.isArray(data.scene_plan) ? data.scene_plan : null;
  return <>
    {characters && <ResolveTable title={`전체 등장인물 (${characters.length}명)`} headers={["순서", "캐릭터"]}
      rows={characters.map((character, index) => [index + 1, characterName(character)])} empty="등록된 등장인물 없음" />}
    {scenes && <ResolveTable title={`선택 장면 (${scenes.length}개)`} headers={["장면", "원문 구간", "등장인물", "장면 요약"]}
      rows={scenes.map((value, index) => {
        const scene = record(value);
        return [index + 1, resultText(scene?.anchor_segment),
          Array.isArray(scene?.characters) ? scene.characters.map(characterName).join("\n") || "등록 인물 없음" : "—",
          resultText(scene?.scene_brief)];
      })} empty="선택된 장면 없음" />}
  </>;
}

function TrackedStateTable({ data, kind }: { data: StateTable; kind: StateKind }) {
  if (!Array.isArray(data.slots) || !Array.isArray(data.characters)) return null;
  const view = stateViews[kind];
  return <section className="illust-flow-results">
    <h3>{view.title}</h3><p className="field-help">{view.help}</p>
    <div className="illust-flow-table-scroll" role="region" aria-label={view.label} tabIndex={0}>
      <table className="illust-flow-result-table" aria-label={view.label}>
        <thead><tr><th scope="col">인물</th>{data.slots.map(slot => <th scope="col" key={slot}>{slotKey(slot)}</th>)}</tr></thead>
        <tbody>{data.characters.map(name => <tr key={name}><th scope="row">{name}</th>{data.slots.map(slot => {
          const key = slotKey(slot), state = data.states?.[key]?.[name] || {}, changes = data.changes?.[key]?.[name] || {};
          return <td key={slot}>{view.fields.map(([field, caption]) => {
            const direct = Object.hasOwn(changes, field);
            return <div className="illust-flow-state-value" key={field} data-field={field} data-source={direct ? "direct" : "inherited"}>
              <strong>{caption} · {direct ? kind === "profile" ? "판정" : "직접" : "이어받음"}</strong>
              <p>{resultText(state[field] || (kind === "outfit" ? "(태그 없음)" : "(정보 없음)"))}</p>
            </div>;
          })}</td>;
        })}</tr>)}
          {!data.characters.length && <tr><td colSpan={data.slots.length + 1}>등록된 등장인물 없음</td></tr>}
        </tbody>
      </table>
    </div>
  </section>;
}

// Match the original flow's task/label aliases and use the server-expanded states.
export function IllustrationFlowResults({ node }: { node: FlowNode }) {
  const callName = String(node.call_name || node.label || "").split(" ")[0];
  if (node.task_key === "illustration_character_resolve" || ["CHARACTER-RESOLVE", "MANGA-CHARACTER-RESOLVE", "CHARACTER-RESOLVE-REPAIR"].includes(callName)) {
    return <CharacterResolveTables output={node.output} />;
  }
  if (node.profile_state_table) return <TrackedStateTable data={node.profile_state_table} kind="profile" />;
  if (node.time_place_state_table) return <TrackedStateTable data={node.time_place_state_table} kind="timePlace" />;
  if (node.call1_state_table) return <TrackedStateTable data={node.call1_state_table} kind="outfit" />;
  return null;
}
