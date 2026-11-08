import {
  useId,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  Box,
  Check,
  ChevronRight,
  Cpu,
  ExternalLink,
  FileBox,
  FolderHeart,
  Info,
  KeyRound,
  Link2,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import {
  Badge,
  CredentialField,
  IconBox,
  Note,
  Panel,
  Toggle,
} from "./components";
import {
  CIVITAI_PROVIDER_ID,
  WORKFLOW_KEY_ID,
  workflowKeyEntry,
  initialDraft,
  type Draft,
} from "./data";
import { InstallControls } from "./InstallControls";
import type { SaveState } from "./useSettings";

type Props = {
  draft: Draft;
  setDraft: Dispatch<SetStateAction<Draft>>;
  notify: (text: string) => void;
  isSaved: boolean;
};


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

export function UpdatesPage({ draft, setDraft, isSaved }: Props) {
  const update = draft.updates;
  function change(patch: Partial<Draft["updates"]>) {
    setDraft((d) => ({
      ...d,
      updates: { ...d.updates, ...patch, saved: false },
    }));
  }
  return (
    <div className="content-grid">
      <div className="stack">
        <section className="update-banner">
          <span className="update-emblem">
            <RefreshCw size={28} strokeWidth={1.5} />
          </span>
          <div>
            <Badge tone="lavender">V6 PREVIEW</Badge>
            <h2>다음 버전을 위한 준비</h2>
            <p>프로그램과 워크플로우가 함께 업데이트되는 환경.</p>
          </div>
        </section>
        <Panel title="업데이트 구성">
          <div className="component-row">
            <IconBox icon={Box} />
            <div>
              <strong>Comfy공방 프로그램</strong>
              <p>설정 · 삽화 · 에셋</p>
            </div>
            <Badge>배포 연결 대기</Badge>
          </div>
          <div className="component-row">
            <IconBox icon={FileBox} tone="lavender" />
            <div>
              <strong>내장 워크플로우</strong>
              <p>프로그램 버전과 맞춰 함께 제공</p>
            </div>
            <Badge>배포 연결 대기</Badge>
          </div>
          <div className="component-row">
            <IconBox icon={Cpu} tone="sage" />
            <div>
              <strong>ComfyUI 실행 환경</strong>
              <p>Python · 커스텀 노드 · 실행 의존성</p>
            </div>
            <Badge>환경 검사 대기</Badge>
          </div>
          <div className="panel-foot">
            <Info size={15} />
            <p>현재 버전과 새 릴리스는 엔진 연결 후 조회합니다.</p>
          </div>
        </Panel>
        <Panel title="업데이트 설정">
          <div className="panel-body">
            <label className="field">
              업데이트 채널
              <select
                value={update.channel}
                onChange={(e) =>
                  change({ channel: e.target.value as "stable" | "preview" })
                }
              >
                <option value="stable">Stable · 안정 버전</option>
                <option value="preview">Preview · 사전 공개 버전</option>
              </select>
            </label>
            <div className="setting-row">
              <div>
                <strong>업데이트 전 백업</strong>
                <p>변경되는 프로그램 파일을 백업하도록 계획합니다.</p>
              </div>
              <Toggle
                label="업데이트 전 백업"
                checked={update.backup}
                onChange={(backup) => change({ backup })}
              />
            </div>
            <div className="setting-row">
              <div>
                <strong>내장 워크플로우 함께 업데이트</strong>
                <p>업데이트가 끝나면 다시 읽기 전용으로 보호합니다.</p>
              </div>
              <Toggle
                label="내장 워크플로우 함께 업데이트"
                checked={update.workflow}
                onChange={(workflow) => change({ workflow })}
              />
            </div>
            <button
              className="button primary"
              onClick={() => {
                setDraft((d) => ({
                  ...d,
                  updates: { ...d.updates, saved: true },
                }));
              }}
            >
              <Check size={15} />
              {update.saved && isSaved
                ? "업데이트 정책 저장됨"
                : "업데이트 정책 저장"}
            </button>
          </div>
        </Panel>
      </div>
      <div className="stack">
        <Panel title="내 데이터는 그대로" aside={<ShieldCheck size={16} />}>
          <div className="panel-body protection-list">
            <FolderHeart size={30} strokeWidth={1.3} />
            <h3>
              작업은 이어지고,
              <br />
              환경은 새로워지도록.
            </h3>
            <p>
              V6 업데이트는 사용자 데이터와 프로그램을 분리하는 구조를 기준으로
              설계합니다.
            </p>
            <span>
              <Check size={14} />
              모델과 LoRA
            </span>
            <span>
              <Check size={14} />
              API 설정과 프리셋
            </span>
            <span>
              <Check size={14} />
              생성한 이미지와 작업물
            </span>
          </div>
        </Panel>
        <Note>
          이 초안은 업데이트 정책만 저장합니다. 버전 확인, 백업 및 실제 교체는
          아직 실행되지 않습니다.
        </Note>
      </div>
    </div>
  );
}

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
