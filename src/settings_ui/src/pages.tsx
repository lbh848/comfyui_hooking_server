import {
  useId,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  ChevronRight,
  ExternalLink,
  KeyRound,
  Link2,
} from "lucide-react";
import {
  Badge,
  CredentialField,
  Panel,
} from "./components";
import {
  CIVITAI_PROVIDER_ID,
  WORKFLOW_KEY_ID,
  workflowKeyEntry,
  initialDraft,
} from "./data";
import { InstallControls } from "./InstallControls";
import type { SaveState } from "./useSettings";

type CredentialProps = {
  apiKeys: Record<string, string>;
  setApiKeys: Dispatch<SetStateAction<Record<string, string>>>;
};

function SetupCredentialFields({
  credentialId,
  apiKeys,
  setApiKeys,
}: CredentialProps & { credentialId: string }) {
  const isCivitai = credentialId === CIVITAI_PROVIDER_ID;
  return (
    <>
      {isCivitai && (
        <a
          className="text-button credential-link"
          href="https://civitai.com/"
          target="_blank"
          rel="noreferrer"
        >
          Civitai 바로가기 <ExternalLink size={14} />
        </a>
      )}
      <CredentialField
        label={isCivitai ? "Civitai API 키" : "암호키"}
        value={apiKeys[credentialId] ?? ""}
        onChange={(value) =>
          setApiKeys((keys) => ({ ...keys, [credentialId]: value }))
        }
      />
      {!isCivitai && (
        <p className="field-help">
          설치할 워크플로우와 프리셋의 복호화에 사용하는 암호키입니다.
        </p>
      )}
    </>
  );
}

function SharedCredentialLink() {
  const tooltipId = useId();
  const [dismissed, setDismissed] = useState(false);
  return (
    <span
      className="credential-link-tooltip"
      data-dismissed={dismissed || undefined}
      onPointerEnter={() => setDismissed(false)}
      onFocus={() => setDismissed(false)}
      onKeyDown={(event) => {
        if (event.key === "Escape") setDismissed(true);
      }}
    >
      <a
        className="icon-button credential-shared-link"
        href="#other-api"
        aria-label="기타 API 관리로 이동"
        aria-describedby={tooltipId}
      >
        <Link2 size={15} aria-hidden="true" />
      </a>
      <span className="credential-tooltip" id={tooltipId} role="tooltip">
        {"리소스 관리 -> 기타 API 관리와 호환되는 메뉴입니다"}
      </span>
    </span>
  );
}

export function InstallPage(props: CredentialProps & { isSaved: boolean; saveState: SaveState; revision: string }) {
  return (
    <div className="stack install-credentials">
      <Panel className="credential-panel" title={<span className="credential-heading">01 · Civitai API 키 <SharedCredentialLink /></span>}>
        <div className="panel-body form-stack">
          <SetupCredentialFields
            {...props}
            credentialId={CIVITAI_PROVIDER_ID}
          />
        </div>
      </Panel>
      <Panel className="credential-panel" title={<span className="credential-heading">02 · 암호키 <SharedCredentialLink /></span>}>
        <div className="panel-body form-stack">
          <SetupCredentialFields {...props} credentialId={WORKFLOW_KEY_ID} />
        </div>
      </Panel>
      <InstallControls
        isSaved={props.isSaved}
        saveState={props.saveState}
        revision={props.revision}
        hasCivitaiKey={!!props.apiKeys[CIVITAI_PROVIDER_ID]?.trim()}
        hasWorkflowKey={!!props.apiKeys[WORKFLOW_KEY_ID]?.trim()}
      />
    </div>
  );
}

export { UpdatesPage } from "./UpdatesPage";

export function OtherApiPage({
  apiKeys,
  setApiKeys,
}: CredentialProps) {
  const providers = [...initialDraft.otherProviders, workflowKeyEntry];
  const [selected, setSelected] = useState(providers[0].id);
  const provider = providers.find((p) => p.id === selected) ?? providers[0];
  return (
    <div className="content-grid api-layout">
      <Panel title="외부 서비스">
        <div className="provider-list">
          {providers.map((p, index) => (
            <button
              className={`provider-option ${provider.id === p.id ? "selected" : ""}`}
              key={p.id}
              aria-pressed={provider.id === p.id}
              onClick={() => setSelected(p.id)}
            >
              <span className={`provider-logo provider-${index}`}>
                <KeyRound size={23} />
              </span>
              <div>
                <strong>{p.name}</strong>
                <small>
                  {p.id === CIVITAI_PROVIDER_ID
                    ? "모델·LoRA 리소스"
                    : "워크플로우·프리셋 복호화"}
                </small>
              </div>
              <Badge tone={apiKeys[p.id] ? "green" : "neutral"}>
                {apiKeys[p.id] ? "입력됨" : "미입력"}
              </Badge>
              <ChevronRight size={15} />
            </button>
          ))}
        </div>
      </Panel>
      <Panel title={provider.name} aside={<KeyRound size={16} />}>
        <div className="panel-body form-stack">
          <SetupCredentialFields
            credentialId={provider.id}
            apiKeys={apiKeys}
            setApiKeys={setApiKeys}
          />
        </div>
      </Panel>
    </div>
  );
}
