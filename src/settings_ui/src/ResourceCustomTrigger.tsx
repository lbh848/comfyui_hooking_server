import { useId, useState } from "react";
import { CircleHelp } from "lucide-react";

const help = "프로그램에서 로드했을 때 사용할 트리거 워드입니다. 여러개의 트리거 워드를 가진 로라를 사용할 경우 주어진 로라 중 하나의 로라만 사용하면 되기 때문에 원하는 트리거만 프롬프트에 작성하도록 두고 있습니다.";

export function ResourceCustomTrigger({ value = "" }: { value?: string }) {
  const id = useId();
  const [dismissed, setDismissed] = useState(false);
  return <div className="field">
    <div className="resource-trigger-field-heading">
      <label htmlFor={id}>커스텀 트리거 (선택)</label>
      <span className="resource-trigger-help" data-dismissed={dismissed || undefined}
        onPointerEnter={() => setDismissed(false)} onFocus={() => setDismissed(false)}
        onKeyDown={event => {
          if (event.key === "Escape" && !dismissed) {
            event.preventDefault(); event.stopPropagation(); setDismissed(true);
          }
        }}>
        <button type="button" className="icon-button" aria-label="커스텀 트리거 도움말" aria-describedby={`${id}-help`}>
          <CircleHelp size={16} aria-hidden="true" />
        </button>
        <span id={`${id}-help`} className="resource-trigger-tooltip" role="tooltip">{help}</span>
      </span>
    </div>
    <textarea id={id} name="customTrigger" rows={2} maxLength={16384} defaultValue={value}
      placeholder="사용할 트리거만 입력하세요 (쉼표 또는 줄바꿈으로 구분)" />
  </div>;
}
