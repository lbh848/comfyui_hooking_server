import { Badge, Empty, Modal } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { useEmotionProfileOutfitGeneration } from "./useEmotionProfileOutfitGeneration";

/** 프로필 카드의 기본 복장 이미지를 한 장씩 미리 생성하는 도구. */
export function EmotionProfileOutfitGeneration({ botName, readOnly, onClose, onChanged }: {
  botName: string; readOnly: boolean; onClose: () => void; onChanged: () => void;
}) {
  const state = useEmotionProfileOutfitGeneration(botName, onChanged);
  const pending = state.plans.filter(plan => plan.action === "generate");
  const skipped = state.plans.length - pending.length;
  const run = state.run;
  const progress = state.active && run ? { done: Number(run.done || 0), total: Number(run.total || 0), current: String(run.current || "") } : null;
  const errors: string[] = Array.isArray(run?.errors) ? run.errors as string[] : [];
  const warnings: string[] = Array.isArray(run?.warnings) ? run.warnings as string[] : [];

  // 생성 중에도 창을 백그라운드로 내릴 수 있다. 모달은 닫히지만 컴포넌트는
  // 계속 마운트되어 폴링/진행 표시가 유지되고, 하단 '백그라운드 작업'에서 복원한다.
  return <Modal title={`${botName} · 프로필별 복장 이미지 생성`} backgroundable working={state.active} onClose={onClose}>
    <div className="illust-form">
      <p className="field-help">각 프로필 카드의 기본 복장·외모 태그와 이 봇의 그리기 설정으로 복장 대표 표정 이미지를 1장씩 생성합니다. <strong className="emotion-profile-outfit-emphasis">에셋 생성 모드</strong>를 활용하여 진행됩니다. 이미 복장 이미지가 있는 프로필은 건너뜁니다.</p>
      <IllustrationFeedback error={state.error} message={state.message} />
      {state.plansLoading && !state.plans.length ? <p className="field-help" role="status">프로필을 확인하는 중…</p>
        : !state.plans.length ? <Empty title="프로필이 없습니다">캐릭터 프로필 카드와 기본 복장 태그를 먼저 등록하세요.</Empty>
          : <>
            <div className="illust-actions"><Badge tone="green">생성 예정 {pending.length}</Badge><Badge>건너뜀 {skipped}</Badge></div>
            <ul className="emotion-profile-outfit-list" aria-label="프로필 복장 이미지 계획">
              {state.plans.map(plan => <li key={`${plan.character}:${plan.profile_id}`} className={plan.action === "generate" ? "is-pending" : "is-skipped"}>
                <span className="emotion-profile-outfit-who"><strong>{plan.character}</strong><span>{plan.profile_label}</span></span>
                <span className="emotion-profile-outfit-what" title={plan.outfit}>{plan.outfit || "기본 복장 없음"}{plan.emotion_name ? ` · ${plan.emotion_name}` : ""}</span>
                <Badge tone={plan.action === "generate" ? "green" : "neutral"}>{plan.action === "generate" ? "생성 예정" : plan.reason || "건너뜀"}</Badge>
              </li>)}
            </ul>
          </>}
      {progress && <p className="field-help" role="status">프로필 복장 생성 중… {progress.done}/{progress.total}{progress.current ? ` · ${progress.current}` : ""}</p>}
      {!!warnings.length && <div className="field-help"><strong>알림</strong><ul>{warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></div>}
      {!!errors.length && <div className="field-error"><strong>실패한 프로필</strong><ul>{errors.map((item, index) => <li key={index}>{item}</li>)}</ul></div>}
      <div className="illust-actions illust-dialog-actions">
        <span className="field-help">작업 큐에서 진행됩니다. 창을 백그라운드로 내려도 생성은 계속됩니다.</span>
        <button type="button" className="button" onClick={onClose}>닫기</button>
        {state.active && <button type="button" className="button danger" disabled={state.cancelling} onClick={() => void state.cancel()}>{state.cancelling ? "중단 중…" : "중단"}</button>}
        <button type="button" className="button primary" disabled={readOnly || state.busy || !pending.length} onClick={() => void state.generate()}>
          {state.busy ? "생성 중…" : `생성 시작 (${pending.length})`}
        </button>
      </div>
    </div>
  </Modal>;
}
