/* Live execution graph; prompt bodies are fetched only when a port is opened. */
(() => {
    let flow = null, flowBackdrop, modal, detailModal, developerModal, stopButton, developerButton;
    let selected = null, detailRequest = 0, developerPreviousFocus = null;
    let detailRefreshRequest = 0, detailRefreshPending = false, renderedDetail = null;
    let qualityInspectionEnabled = false, qualityInspectionRequest = 0;
    let scale = 1, previousFocus = null;
    let activeTab = 'illustration';
    const flows = {illustration: null, video: null, video_input: null};
    const views = {
        illustration: {scale: 1, scroll: [0, 0]},
        video: {scale: 1, scroll: [0, 0]},
        video_input: {scale: 1, scroll: [0, 0]},
    };
    const tabLabels = {
        illustration: '삽화 흐름 보기',
        video: '영상 흐름 보기',
        video_input: '영상 입력 개선',
    };
    const autoOpenStorageKey = kind => `workflow-flow-auto-open-${kind}`;
    const autoOpenEnabled = {illustration: true, video: true, video_input: false};
    Object.keys(autoOpenEnabled).forEach(kind => {
        try {
            const saved = localStorage.getItem(autoOpenStorageKey(kind));
            if (saved !== null) autoOpenEnabled[kind] = saved === 'true';
        } catch (error) {
            console.error('[ILLUST_FLOW] 자동 열기 설정 조회 실패:', {kind, enabled: autoOpenEnabled[kind]}, error);
        }
    });
    const flowUrl = kind => kind === 'illustration'
        ? '/api/illustration_flow'
        : `/api/illustration_flow?kind=${encodeURIComponent(kind)}`;
    const qualityInspectionSettingsUrl = '/api/illustration_quality_inspection/settings';
    const labels = {waiting: '대기', processing: '처리 중', cancelling: '중단 중', completed: '완료', failed: '실패', cancelled: '취소', skipped: '생략'};
    const colors = {waiting: '#94a3b8', processing: '#60a5fa', cancelling: '#f59e0b', completed: '#4ade80', failed: '#fb7185', cancelled: '#fbbf24', skipped: '#a78bfa'};
    const executorLabels = {llm: 'LLM', comfy: 'Comfy', human: '사람', process: '기타 프로세스'};
    const executorColors = {llm: '#a78bfa', comfy: '#22d3ee', human: '#f59e0b', process: '#94a3b8'};
    const nodeExecutor = n => n.executor || (n.kind === 'llm' ? 'llm' : 'process');
    const element = (tag, className, text) => {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (text !== undefined) el.textContent = text;
        return el;
    };
    const stringify = value => typeof value === 'string' ? value : JSON.stringify(value ?? '', null, 2);
    const updateText = (el, text) => {if (el.textContent !== text) el.textContent = text;};
    const hasDetailValue = value => value !== null && value !== undefined && (typeof value !== 'string' || value.trim() !== '');
    const maskResults = new Map();
    const previewImages = new Map();
    const maskOverlayNodes = new Set();
    let maskResultsRun = '';
    function detailOutput(n) {
        if (hasDetailValue(n.output)) return n.output;
        const attempts = Array.isArray(n.attempts) ? n.attempts : [];
        let latestRawResponse;
        for (let index = attempts.length - 1; index >= 0; index -= 1) {
            if (attempts[index] && hasDetailValue(attempts[index].raw_response)) {
                latestRawResponse = attempts[index].raw_response;
                break;
            }
        }
        const fallback = {};
        if (hasDetailValue(n.error)) fallback.error = n.error;
        if (hasDetailValue(latestRawResponse)) fallback.raw_response = latestRawResponse;
        return Object.keys(fallback).length ? fallback : '';
    }
    const elapsed = n => n.started_at ? `${Math.max(0, (n.ended_at || Date.now() / 1000) - n.started_at).toFixed(1)}초` : n.ended_at ? '처리 종료' : '실행 대기';
    const phaseSeconds = (n, phase) => {
        const start = n[`${phase}_started_at`];
        if (!Number.isFinite(start)) return null;
        const active = n.active === true || n.status === 'processing' || n.status === 'cancelling';
        const end = phase === 'thinking' && Number.isFinite(n.response_started_at)
            ? n.response_started_at
            : active ? Date.now() / 1000
            : phase === 'thinking' ? n.last_event_at ?? n.ended_at ?? n.thinking_ended_at
            : n.response_ended_at;
        return Number.isFinite(end) ? Math.max(0, end - start) : null;
    };
    function llmTiming(n) {
        if (nodeExecutor(n) !== 'llm') return '';
        const thinkingSeconds = phaseSeconds(n, 'thinking');
        const thinking = thinkingSeconds === null
            ? (n.active === true || n.status === 'processing' || n.status === 'waiting') && !Number.isFinite(n.response_started_at) ? '대기' : '미수신'
            : thinkingSeconds < 0.1 ? '<0.1s' : `${thinkingSeconds.toFixed(1)}s`;
        const responseSeconds = phaseSeconds(n, 'response');
        const response = Number.isFinite(n.response_started_at)
            ? responseSeconds === null ? '측정 불가' : responseSeconds < 0.1 ? '<0.1s' : `${responseSeconds.toFixed(1)}s`
            : n.status === 'completed' ? '일괄'
            : ['failed', 'cancelled'].includes(n.status) || Number.isFinite(n.ended_at) ? '미측정' : '대기';
        return `THINK ${thinking} · 응답 ${response}`;
    }
    const tokenTime = value => Number.isFinite(value)
        ? new Date(value * 1000).toLocaleTimeString('ko-KR', {
            hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3,
        }) : '수신 없음';
    function timingDetails(n) {
        if (nodeExecutor(n) !== 'llm') return {};
        return {
            'THINK 시간': phaseSeconds(n, 'thinking') === null ? '측정 불가 (추론 수신 없음)' : `${phaseSeconds(n, 'thinking').toFixed(1)}s`,
            '응답 작성 시간': phaseSeconds(n, 'response') === null ? '토큰별 수신 기록 없음' : `${phaseSeconds(n, 'response').toFixed(1)}s`,
            '추론 첫 수신': tokenTime(n.thinking_started_at),
            '추론 마지막 수신': tokenTime(n.thinking_ended_at),
            '본문 첫 수신': tokenTime(n.response_started_at),
            '본문 마지막 수신': tokenTime(n.response_ended_at),
        };
    }
    function button(text, action) {
        const b = element('button', 'if-button', text); b.type = 'button'; b.onclick = action; return b;
    }
    function init() {
        if (modal) return;
        const style = element('style');
        style.textContent = `
            /* z-index 표준 래더(index.html 상단 참조): 460 백드롭 < 470 모달 < 480 상세 < 490 개발자 */
            .if-layer-backdrop{position:fixed;inset:0;z-index:460;background:#020617b3;backdrop-filter:blur(3px)}
            .if-layer-backdrop[hidden]{display:none}
            .if-modal{position:fixed;inset:0;z-index:470;margin:auto;box-sizing:border-box;overflow:hidden;color:var(--text,#e2e8f0);background:var(--bg2,#111827);border:1px solid #64748b66;border-radius:16px;padding:0;width:min(1240px,94vw);max-width:96vw;max-height:calc(100dvh - 32px);box-shadow:0 24px 90px #0009;font:14px/1.5 system-ui,sans-serif}
            .if-modal[open]{display:flex;flex-direction:column}.if-header,.if-tabs,.if-auto-open-settings,.if-legend,.if-footer{flex-shrink:0}
            .if-detail{z-index:480}
            .if-call1-open{border:0;background:transparent;color:inherit;text-align:left;cursor:pointer;padding:0;width:100%;font:inherit}.if-call1-open:focus-visible{outline:2px solid #60a5fa;outline-offset:3px}
            .if-state-table-wrap{overflow:auto;max-height:55vh;border:1px solid #64748b55;border-radius:10px;margin:12px 0}.if-state-table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-size:12px}.if-state-table th,.if-state-table td{border-right:1px solid #64748b44;border-bottom:1px solid #64748b44;padding:10px;vertical-align:top}.if-state-table thead th{position:sticky;top:0;background:var(--bg2,#111827);z-index:2}.if-state-table tbody th{position:sticky;left:0;background:var(--bg2,#111827);z-index:1;min-width:90px}.if-state-table td{min-width:190px;max-width:260px;overflow-wrap:anywhere}.if-state-tags+.if-state-tags{margin-top:8px}.if-state-direct{color:#93c5fd}.if-state-inherited{color:var(--text2,#94a3b8)}.if-state-tag-label{font-weight:700;font-size:11px;margin-bottom:3px}
            .if-developer{z-index:490;width:min(620px,94vw)}
            .if-header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:18px 22px;border-bottom:1px solid #64748b44}
            .if-header h2{font-size:19px;margin:0}.if-title-row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.if-subtitle{font-size:12px;color:var(--text2,#94a3b8);margin-top:4px;overflow-wrap:anywhere}
            .if-actions,.if-legend,.if-legend-group{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.if-button{border:1px solid #64748b66;border-radius:8px;background:transparent;color:inherit;padding:6px 11px;cursor:pointer}.if-button:hover{background:#64748b33}.if-button:disabled{opacity:.45;cursor:not-allowed}.if-button-danger{border-color:#fb718580;color:#fecdd3}.if-button-danger:not(:disabled):hover{background:#fb71851f}.if-button:focus-visible,.if-port:focus-visible{outline:3px solid #60a5fa;outline-offset:3px}
            .if-developer-button{border-color:#c084fc99;color:#e9d5ff}.if-developer-button:hover{background:#c084fc1f}
            .if-tabs{display:flex;gap:8px;padding:12px 22px 0;border-bottom:1px solid #64748b44}.if-tab{border:0;border-bottom:3px solid transparent;border-radius:8px 8px 0 0;padding:10px 16px;color:var(--text2,#94a3b8)}.if-tab[aria-selected=true]{color:var(--text,#e2e8f0);background:#60a5fa18;border-bottom-color:#60a5fa}.if-modal [hidden]{display:none!important}
            .if-auto-open-settings{padding:10px 22px;border-bottom:1px solid #64748b33}.if-auto-open-toggle{display:flex;align-items:center;gap:9px;width:fit-content;font-size:13px;cursor:pointer}.if-auto-open-toggle input{appearance:none;flex-shrink:0;width:34px;height:20px;margin:0;border:1px solid #64748b;border-radius:12px;background:#475569;cursor:pointer;position:relative}.if-auto-open-toggle input::after{content:'';position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;background:#e2e8f0;transition:transform .15s}.if-auto-open-toggle input:checked{background:#2563eb;border-color:#60a5fa}.if-auto-open-toggle input:checked::after{transform:translateX(14px)}.if-auto-open-toggle input:focus-visible{outline:3px solid #60a5fa;outline-offset:3px}.if-auto-open-state{font-size:12px;color:var(--text2,#94a3b8)}.if-auto-open-help{font-size:12px;color:var(--text2,#94a3b8);margin:5px 0 0;overflow-wrap:anywhere}
            .if-legend{padding:10px 22px;gap:18px;font-size:12px;border-bottom:1px solid #64748b33}.if-legend-group{gap:12px}.if-legend-label{color:var(--text2,#94a3b8);font-weight:650}.if-legend-status .if-legend-item::before{content:'●';color:var(--state);margin-right:5px}.if-legend-executor .if-legend-item::before{content:'';display:inline-block;width:13px;height:13px;border-radius:4px;background:color-mix(in srgb,var(--bg2,#172033) 68%,var(--node-tint) 32%);border:1px solid var(--node-tint);margin-right:6px;vertical-align:-2px}
            .if-viewport{height:min(65vh,660px);min-height:0;flex-shrink:1;overflow:auto;background-color:var(--bg,#0b1220);background-image:radial-gradient(#94a3b822 1px,transparent 1px);background-size:20px 20px;padding:0;position:relative}
            .if-space{position:relative}.if-canvas{position:relative;transform-origin:0 0}.if-edges{position:absolute;inset:0;overflow:visible;pointer-events:none}
            .if-direction{position:absolute;left:36px;top:12px;color:var(--text2,#94a3b8);font-size:12px}.if-step-column{position:absolute;top:44px;box-sizing:border-box;width:246px;border:1px solid #64748b33;border-radius:10px;background:#64748b06;pointer-events:none}.if-step-title{padding:8px 12px;border-bottom:1px solid #64748b33;color:var(--text2,#94a3b8);font-size:12px;font-weight:750}
            .if-node{position:absolute;box-sizing:border-box;width:222px;height:98px;border:1px solid #64748b77;border-left:4px solid var(--state);border-radius:11px;background:color-mix(in srgb,var(--bg2,#172033) 68%,var(--node-tint,#94a3b8) 32%);padding:12px 22px 10px 13px;box-shadow:0 4px 15px #0002}
            .if-node[data-status=processing]{box-shadow:0 0 0 2px #60a5fa33,0 0 22px #60a5fa22}.if-node-title{font-weight:650;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.if-node-state{color:var(--state);font-size:12px;margin-top:6px}.if-node-model{font-size:11px;color:var(--text2,#94a3b8);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
            .if-node-heading{display:flex;align-items:center;gap:6px;min-width:0}.if-node-heading .if-node-title{flex:1;min-width:0}.if-node-attempts{flex-shrink:0;white-space:nowrap;font-size:11px;font-weight:700;line-height:1.5;padding:1px 5px;border:1px solid #c4b5fd99;border-radius:6px;background:#a78bfa26;color:var(--text,#e2e8f0)}
            .if-port{position:absolute;right:-9px;top:39px;width:18px;height:18px;border-radius:50%;border:3px solid color-mix(in srgb,var(--bg2,#172033) 68%,var(--node-tint,#94a3b8) 32%);background:var(--state);cursor:pointer;padding:0;box-shadow:0 0 0 1px var(--state)}
            .if-node-timing{font-size:10px;line-height:1.3;color:var(--text2,#cbd5e1);white-space:nowrap;margin-top:2px}
            .if-node-preview{position:absolute;top:98px;left:13px;right:13px;bottom:12px;overflow:auto;border-top:1px solid #64748b55;padding-top:8px;font-size:11px;overflow-wrap:anywhere}.if-preview-title{font-weight:700;margin-bottom:5px}.if-preview-empty{padding:38px 5px;text-align:center;color:var(--text2,#94a3b8)}.if-mask-preview{display:block;width:100%;height:130px;background:#020617;border-radius:6px}.if-mask-region{margin-top:3px;line-height:1.4}.if-mask-text{white-space:pre-wrap;font:10px/1.5 system-ui;margin:7px 0 0;max-height:65px;overflow:auto;color:var(--text2,#cbd5e1)}.if-preview-image-button{display:block;width:100%;padding:0;border:0;background:#020617;border-radius:6px;cursor:pointer}.if-preview-image{display:block;width:100%;height:190px;object-fit:contain;border-radius:6px}.if-detail-image{display:block;width:100%;max-height:540px;object-fit:contain;background:#020617;border-radius:8px;margin:12px 0}
            .if-mask-overlay-toggle{display:flex;align-items:center;gap:6px;margin-bottom:7px;cursor:pointer}.if-mask-overlay-toggle input{margin:0;accent-color:#60a5fa}.if-mask-overlay-toggle input:focus-visible{outline:2px solid #60a5fa;outline-offset:2px}.if-mask-overlay-toggle:has(input:disabled){color:var(--text2,#94a3b8);cursor:default}.if-preview-image-button{position:relative;isolation:isolate}.if-image-mask-overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;border-radius:6px}
            .if-node[data-review-node=true] .if-port{width:22px;height:22px;right:-11px;top:37px;border-color:#fef3c7;background:#f59e0b;box-shadow:0 0 0 2px #f59e0b,0 0 16px #f59e0b99}.if-node[data-review-node=true] .if-node-model{color:#fde68a;font-weight:700}
            .if-tooltip{position:fixed;z-index:30;max-width:330px;padding:10px 13px;border-radius:9px;background:#0f172a;color:#e2e8f0;border:1px solid #64748b;box-shadow:0 8px 30px #0006;white-space:pre-wrap;pointer-events:none;font-size:12px}
            .if-empty{padding:80px 24px;text-align:center;color:var(--text2,#94a3b8)}.if-empty small{display:block;margin-top:8px}.if-footer{padding:10px 22px;font-size:12px;color:var(--text2,#94a3b8)}
            .if-detail{width:min(900px,94vw)}.if-detail-body{padding:18px 22px;max-height:70vh;overflow:auto}.if-detail-body h3{font-size:14px;margin:18px 0 8px}.if-detail-body pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#64748b14;padding:14px;border-radius:8px;font:12px/1.65 ui-monospace,monospace;margin:0;max-height:340px;overflow:auto}.if-detail-body summary{cursor:pointer;padding:8px 0}.if-meta{display:grid;grid-template-columns:110px 1fr;gap:7px 14px;overflow-wrap:anywhere}.if-meta dt{color:var(--text2,#94a3b8)}.if-meta dd{margin:0}
            .if-review-panel{border:1px solid #f59e0b88;border-radius:12px;background:#f59e0b0f;padding:16px;margin-bottom:12px}.if-review-title{font-weight:800;font-size:15px;color:#fde68a}.if-review-guide{margin:5px 0 14px;color:var(--text2,#94a3b8);font-size:12px;line-height:1.65}.if-review-list{display:grid;gap:14px}.if-review-item{display:grid;grid-template-columns:minmax(220px,.9fr) minmax(300px,1.1fr);gap:14px;padding:12px;border:1px solid #64748b55;border-radius:12px;background:#02061755}.if-review-item-body{min-width:0}.if-review-figure{margin:0;border:1px solid #64748b55;border-radius:10px;overflow:hidden;background:#02061766;align-self:start}.if-review-image{display:block;width:100%;max-height:520px;object-fit:contain;background:#020617}.if-review-caption{padding:7px 9px;font-size:11px;color:var(--text2,#94a3b8);overflow-wrap:anywhere}.if-review-missing{padding:44px 12px;text-align:center;color:#fecaca;font-size:12px}.if-review-auto{margin:0 0 10px}.if-review-auto summary{color:#bae6fd;font-size:12px}.if-review-auto pre{margin-top:6px!important;max-height:150px!important}.if-review-ratings{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px}.if-review-rating{min-width:86px;font-weight:750}.if-review-rating[aria-pressed=true][data-rating=good]{border-color:#4ade80;background:#22c55e33;color:#bbf7d0}.if-review-rating[aria-pressed=true][data-rating=normal]{border-color:#facc15;background:#eab30833;color:#fef08a}.if-review-rating[aria-pressed=true][data-rating=bad]{border-color:#fb7185;background:#e11d4833;color:#fecdd3}.if-review-reason{box-sizing:border-box;width:100%;min-height:78px;resize:vertical;border:1px solid #64748b66;border-radius:8px;background:var(--bg,#0b1220);color:var(--text,#e2e8f0);padding:10px 12px;font:13px/1.5 system-ui,sans-serif}.if-review-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:14px}.if-review-save{border-color:#f59e0b99;background:#f59e0b22;font-weight:750}.if-review-status{font-size:12px;color:var(--text2,#94a3b8)}
            .if-developer-body{padding:20px 22px;max-height:70vh;overflow:auto}.if-quality-card{border:1px solid #64748b55;border-radius:10px;padding:15px;background:#64748b12}.if-quality-toggle{display:flex;align-items:center;gap:10px;font-weight:700;cursor:pointer}.if-quality-toggle input{width:18px;height:18px;accent-color:#c084fc}.if-quality-copy{margin:9px 0 0;color:var(--text2,#94a3b8);font-size:12px;line-height:1.65}.if-quality-status{margin-top:12px;min-height:1.4em;color:var(--text2,#94a3b8);font-size:12px}.if-developer-footer{display:flex;justify-content:flex-end;padding:12px 22px;border-top:1px solid #64748b44}
            @media(max-width:650px){.if-header{align-items:flex-start;padding:14px}.if-actions{justify-content:flex-end}.if-header h2{font-size:16px}.if-meta{grid-template-columns:80px 1fr}.if-review-item{grid-template-columns:1fr}}
            .if-attempt{margin:10px 0;border:1px solid #64748b66;border-radius:10px;background:#64748b0c;overflow:hidden}
            .if-detail-body .if-attempt>summary{display:flex;align-items:center;gap:10px;padding:12px 14px;list-style:none;font-weight:650}
            .if-attempt>summary::-webkit-details-marker{display:none}
            .if-attempt>summary::before{content:'▶';flex-shrink:0;font-size:10px;color:var(--text2,#94a3b8)}
            .if-attempt>summary::after{content:'펼치기';margin-left:auto;flex-shrink:0;font-size:11px;font-weight:400;color:var(--text2,#94a3b8)}
            .if-attempt>summary:hover{background:#64748b14}.if-attempt>summary:focus-visible{outline:2px solid #60a5fa;outline-offset:-3px}
            .if-attempt[open]{border-color:#60a5fa99;background:#60a5fa08}
            .if-attempt[open]>summary{background:#60a5fa1a;border-bottom:1px solid #60a5fa44}
            .if-attempt[open]>summary::before{content:'▼';color:#60a5fa}.if-attempt[open]>summary::after{content:'접기';color:#60a5fa}
            .if-attempt-title{min-width:0;overflow-wrap:anywhere}.if-attempt-content{padding:0 14px 14px}
            .if-detail-body .if-attempt-content h3{margin:14px 0 8px}
        `;
        document.head.append(style);
        flowBackdrop = element('div', 'if-layer-backdrop'); flowBackdrop.hidden = true;
        modal = element('dialog', 'if-modal'); modal.id = 'illustration-flow-modal';
        modal.setAttribute('aria-labelledby', 'if-title');
        const header = element('header', 'if-header'), titleBox = element('div');
        const titleRow = element('div', 'if-title-row');
        const title = element('h2', '', '작업 흐름 보기'); title.id = 'if-title';
        developerButton = button('개발자 모드', openDeveloperMode);
        developerButton.classList.add('if-developer-button');
        developerButton.id = 'if-developer-mode';
        developerButton.title = '삽화 생성 이미지 자동 검사 설정';
        titleRow.append(title, developerButton);
        titleBox.append(titleRow, element('div', 'if-subtitle', '최신 요청의 실행 상태'));
        const actions = element('div', 'if-actions');
        stopButton = button('중단', cancelCurrentFlow);
        stopButton.classList.add('if-button-danger');
        stopButton.id = 'if-stop';
        stopButton.title = '현재 삽화 처리 흐름만 중단합니다.';
        const resetZoom = button('100%', () => {scale = 1; updateZoomDisplay(); render();});
        resetZoom.id = 'if-zoom-reset';
        resetZoom.title = '현재 확대 비율을 100%로 되돌립니다.';
        actions.append(stopButton, button('−', () => zoom(-0.15)), button('+', () => zoom(0.15)), resetZoom, button('닫기', () => modal.close()));
        header.append(titleBox, actions);
        const tabs = element('div', 'if-tabs');
        tabs.setAttribute('role', 'tablist'); tabs.setAttribute('aria-label', '작업 흐름 종류');
        const autoOpenSettings = element('div', 'if-auto-open-settings');
        Object.entries(tabLabels).forEach(([key, label]) => {
            const tab = button(label, () => {switchTab(key); void refresh(key);});
            tab.classList.add('if-tab'); tab.id = `if-tab-${key}`;
            tab.setAttribute('role', 'tab'); tab.setAttribute('aria-controls', 'if-flow-panel');
            tab.dataset.tab = key;
            tab.addEventListener('keydown', event => {
                if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
                event.preventDefault();
                const keys = Object.keys(tabLabels);
                const currentIndex = keys.indexOf(key);
                const nextIndex = event.key === 'Home'
                    ? 0
                    : event.key === 'End'
                        ? keys.length - 1
                        : (currentIndex + (event.key === 'ArrowRight' ? 1 : -1) + keys.length) % keys.length;
                const next = keys[nextIndex];
                switchTab(next); modal.querySelector(`#if-tab-${next}`).focus(); void refresh(next);
            });
            tabs.append(tab);
            const setting = element('div', 'if-auto-open-setting');
            setting.dataset.tab = key;
            const toggle = element('label', 'if-auto-open-toggle');
            const checkbox = element('input');
            checkbox.type = 'checkbox'; checkbox.id = `if-auto-open-${key}`;
            checkbox.setAttribute('role', 'switch');
            checkbox.setAttribute('aria-label', `${label} 자동 열기`);
            checkbox.setAttribute('aria-describedby', 'if-auto-open-help');
            checkbox.checked = autoOpenEnabled[key];
            const state = element('span', 'if-auto-open-state', checkbox.checked ? '켜짐' : '꺼짐');
            checkbox.addEventListener('change', () => {
                autoOpenEnabled[key] = checkbox.checked;
                state.textContent = checkbox.checked ? '켜짐' : '꺼짐';
                try {
                    localStorage.setItem(autoOpenStorageKey(key), String(checkbox.checked));
                } catch (error) {
                    console.error('[ILLUST_FLOW] 자동 열기 설정 저장 실패:', {kind: key, enabled: checkbox.checked}, error);
                    if (typeof showToast === 'function') showToast('자동 열기 설정을 저장하지 못했습니다. 현재 페이지에서만 적용됩니다.', 'error');
                }
            });
            toggle.append(checkbox, element('span', '', '이 탭의 새 작업 시작 시 자동으로 열기'), state);
            setting.append(toggle); autoOpenSettings.append(setting);
        });
        const autoOpenHelp = element('p', 'if-auto-open-help', '꺼져 있어도 작업 흐름은 계속 기록되며, 작업 흐름 보기 버튼으로 열어 확인할 수 있습니다.');
        autoOpenHelp.id = 'if-auto-open-help'; autoOpenSettings.append(autoOpenHelp);
        const legend = element('div', 'if-legend');
        const statusLegend = element('div', 'if-legend-group if-legend-status');
        statusLegend.append(element('span', 'if-legend-label', '상태'));
        Object.entries(labels).forEach(([key, label]) => {const s = element('span', 'if-legend-item', label); s.style.setProperty('--state', colors[key]); statusLegend.append(s);});
        const executorLegend = element('div', 'if-legend-group if-legend-executor');
        executorLegend.append(element('span', 'if-legend-label', '처리 주체'));
        Object.entries(executorLabels).forEach(([key, label]) => {const s = element('span', 'if-legend-item', label); s.style.setProperty('--node-tint', executorColors[key]); executorLegend.append(s);});
        legend.append(statusLegend, executorLegend);
        const viewport = element('div', 'if-viewport'); viewport.id = 'if-flow-panel';
        viewport.setAttribute('role', 'tabpanel'); viewport.tabIndex = 0;
        modal.append(header, tabs, autoOpenSettings, legend, viewport, element('footer', 'if-footer', '출력 ●을 클릭하면 상세 내용을 볼 수 있습니다. 생성 이미지 전체 품질 검사 노드의 주황색 출력 ●에서는 모든 이미지를 한 화면에서 평가할 수 있습니다.'));
        modal.addEventListener('close', () => {
            hideTooltip();
            if (developerModal?.open) developerModal.close();
            flowBackdrop.hidden = true;
            rehomeToast();
            previousFocus?.focus();
        });
        let ticker;
        modal.addEventListener('close', () => clearInterval(ticker));
        modal.addEventListener('if-open', () => {
            rehomeToast();
            clearInterval(ticker);
            ticker = setInterval(() => {
                const nodes = new Map((flow?.nodes || []).map(n => [n.id, n]));
                modal.querySelectorAll('.if-node').forEach(card => {
                    const n = nodes.get(card.dataset.nodeId);
                    if (n) {
                        updateText(card.querySelector('.if-node-state'), nodeState(n));
                        const timing = card.querySelector('.if-node-timing');
                        if (timing) updateText(timing, llmTiming(n));
                    }
                });
                const detailNode = nodes.get(selected);
                if (detailModal?.open && detailNode) {
                    const values = timingDetails(detailNode);
                    detailModal.querySelectorAll('[data-detail-timing]').forEach(item => updateText(item, values[item.dataset.detailTiming]));
                }
            }, 1000);
        });
        detailModal = element('dialog', 'if-modal if-detail'); detailModal.setAttribute('aria-label', '처리 단계 상세');
        const dh = element('header', 'if-header'); dh.append(element('h2', '', '처리 단계 상세'), button('닫기', () => detailModal.close()));
        detailModal.append(dh, element('div', 'if-detail-body'));
        detailModal.addEventListener('close', () => {
            // close is queued: a node may already have reopened this dialog.
            if (detailModal.open) return;
            selected = null; detailRequest++; detailRefreshRequest = 0;
            detailRefreshPending = false; renderedDetail = null; rehomeToast();
        });

        developerModal = element('dialog', 'if-modal if-developer');
        developerModal.id = 'illustration-quality-settings-modal';
        developerModal.setAttribute('aria-labelledby', 'if-quality-title');
        const developerHeader = element('header', 'if-header');
        const developerTitle = element('h2', '', '개발자 모드');
        developerTitle.id = 'if-quality-title';
        const developerTitleBox = element('div');
        developerTitleBox.append(developerTitle);
        developerHeader.append(developerTitleBox);
        developerHeader.append(button('닫기', () => developerModal.close()));
        const developerBody = element('div', 'if-developer-body');
        const qualityCard = element('section', 'if-quality-card');
        const qualityLabel = element('label', 'if-quality-toggle');
        const qualityCheckbox = document.createElement('input');
        qualityCheckbox.type = 'checkbox';
        qualityCheckbox.id = 'if-quality-inspection-enabled';
        qualityCheckbox.checked = false;
        qualityCheckbox.addEventListener('change', event => {
            void setQualityInspectionState(Boolean(event.currentTarget.checked));
        });
        qualityLabel.append(qualityCheckbox, element('span', '', '생성 이미지 자동 검사 (ON/OFF)'));
        qualityCard.append(qualityLabel, element('p', 'if-quality-copy', '서버를 켤 때마다 꺼진 상태로 시작합니다. 켜면 생성 이미지마다 문제 중심의 영어 피드백을 남기고, 전체 이미지의 복장·스토리 일관성을 함께 검토합니다. 손 문제는 평가하지 않습니다. 검사 기록은 LLM 로그에만 저장되며, LLM 흐름과 LB Details에서 볼 수 있습니다.'));
        const qualityStatus = element('div', 'if-quality-status', '꺼짐 · 서버를 켤 때마다 꺼진 상태로 시작합니다.');
        qualityStatus.id = 'if-quality-inspection-status';
        qualityCard.append(qualityStatus);
        developerBody.append(qualityCard);
        const developerFooter = element('footer', 'if-developer-footer');
        developerFooter.append(button('닫기', () => developerModal.close()));
        developerModal.append(developerHeader, developerBody, developerFooter);
        developerModal.addEventListener('close', () => {
            qualityInspectionRequest += 1;
            rehomeToast();
            developerPreviousFocus?.focus();
            developerPreviousFocus = null;
        });
        document.body.append(flowBackdrop, modal, detailModal, developerModal);
        [modal, detailModal, developerModal].forEach(dialog => dialog.addEventListener('keydown', event => {
            if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); dialog.close(); }
            else if (event.key === 'Tab') event.stopPropagation();
        }));
    }
    function nodeState(n) {
        const progress = n.status === 'processing' && Number.isFinite(n.progress) ? ` · ${Math.round(n.progress)}%` : '';
        return `${labels[n.status] || n.status}${progress} · ${elapsed(n)}`;
    }
    function switchTab(kind) {
        init();
        if (kind !== activeTab) {
            const viewport = modal.querySelector('.if-viewport');
            views[activeTab] = {scale, scroll: [viewport.scrollLeft, viewport.scrollTop]};
            if (detailModal.open) detailModal.close();
            if (developerModal.open) developerModal.close();
            activeTab = kind; flow = flows[kind]; scale = views[kind].scale;
            updateZoomDisplay(); render();
            [viewport.scrollLeft, viewport.scrollTop] = views[kind].scroll;
        } else render();
    }
    function updateStopButton() {
        if (!stopButton) return;
        stopButton.hidden = activeTab !== 'illustration';
        developerButton.hidden = activeTab !== 'illustration';
        const terminal = ['completed', 'failed', 'cancelled', 'skipped'].includes(flow?.status);
        const cancelling = Boolean(flow?.cancel_requested) || flow?.status === 'cancelling';
        stopButton.disabled = !flow || terminal || cancelling;
        stopButton.textContent = !terminal && cancelling ? '중단 중…' : '중단';
    }
    function qualityInspectionToast(message, type = 'info') {
        if (typeof showToast === 'function') showToast(message, type);
        else console.error(`[ILLUST_FLOW] 토스트 표시 불가: ${message}`);
    }
    function setQualityInspectionStatus(message, tone = '') {
        const status = developerModal?.querySelector('#if-quality-inspection-status');
        if (!status) return;
        status.textContent = message;
        status.style.color = tone === 'error'
            ? '#fecdd3'
            : tone === 'success'
                ? '#bbf7d0'
                : '';
    }
    async function loadQualityInspectionSettings() {
        const checkbox = developerModal?.querySelector('#if-quality-inspection-enabled');
        if (!checkbox) return;
        const request = ++qualityInspectionRequest;
        checkbox.disabled = true;
        setQualityInspectionStatus('설정을 불러오는 중…');
        try {
            const response = await fetch(qualityInspectionSettingsUrl, {cache: 'no-store'});
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) throw Error(payload.error || `설정 조회 실패 (${response.status})`);
            if (typeof payload.enabled !== 'boolean') {
                throw Error('설정 조회 응답에 enabled 값이 없습니다.');
            }
            if (request !== qualityInspectionRequest || !developerModal?.open) return;
            qualityInspectionEnabled = payload.enabled;
            checkbox.checked = qualityInspectionEnabled;
            setQualityInspectionStatus(qualityInspectionEnabled
                ? '켜짐 · 다음 삽화 생성부터 자동 검사를 요청합니다.'
                : '꺼짐 · 서버를 켤 때마다 꺼진 상태로 시작합니다.');
        } catch (error) {
            console.error('[ILLUST_QUALITY] 설정 조회 실패:', error);
            if (request !== qualityInspectionRequest || !developerModal?.open) return;
            checkbox.checked = qualityInspectionEnabled;
            setQualityInspectionStatus(`설정 조회 실패: ${error.message || error}`, 'error');
            qualityInspectionToast(`삽화 품질 검사 설정 조회 실패: ${error.message || error}`, 'error');
        } finally {
            if (request === qualityInspectionRequest && developerModal?.open) checkbox.disabled = false;
        }
    }
    async function setQualityInspectionState(nextValue) {
        const checkbox = developerModal?.querySelector('#if-quality-inspection-enabled');
        if (!checkbox) return;
        const previousValue = qualityInspectionEnabled;
        const request = ++qualityInspectionRequest;
        qualityInspectionEnabled = Boolean(nextValue);
        checkbox.disabled = true;
        setQualityInspectionStatus('실행 상태를 변경하는 중…');
        try {
            const response = await fetch(qualityInspectionSettingsUrl, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: qualityInspectionEnabled}),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) throw Error(payload.error || `실행 상태 변경 실패 (${response.status})`);
            if (typeof payload.enabled !== 'boolean') {
                throw Error('실행 상태 변경 응답에 enabled 값이 없습니다.');
            }
            if (request !== qualityInspectionRequest || !developerModal?.open) return;
            qualityInspectionEnabled = payload.enabled;
            checkbox.checked = qualityInspectionEnabled;
            setQualityInspectionStatus(qualityInspectionEnabled
                ? '켜짐 · 다음 삽화 생성부터 자동 검사를 요청합니다.'
                : '꺼짐 · 서버를 켤 때마다 꺼진 상태로 시작합니다.', 'success');
            qualityInspectionToast(`삽화 품질 자동 검사 ${qualityInspectionEnabled ? '켜짐' : '꺼짐'}`, 'success');
        } catch (error) {
            console.error('[ILLUST_QUALITY] 실행 상태 변경 실패:', {nextValue, error});
            if (request !== qualityInspectionRequest || !developerModal?.open) return;
            qualityInspectionEnabled = previousValue;
            checkbox.checked = previousValue;
            setQualityInspectionStatus(`변경 실패 · ${previousValue ? '켜짐' : '꺼짐'}으로 되돌렸습니다.`, 'error');
            qualityInspectionToast(`삽화 품질 검사 실행 상태 변경 실패: ${error.message || error}`, 'error');
        } finally {
            if (request === qualityInspectionRequest && developerModal?.open) checkbox.disabled = false;
        }
    }
    function openDeveloperMode() {
        init();
        if (!developerModal.open) {
            developerPreviousFocus = document.activeElement;
            const checkbox = developerModal.querySelector('#if-quality-inspection-enabled');
            checkbox.checked = qualityInspectionEnabled;
            developerModal.show();
            rehomeToast();
            checkbox.focus({preventScroll: true});
        }
        void loadQualityInspectionSettings();
    }
    async function cancelCurrentFlow() {
        const runId = flow?.id;
        if (!runId || stopButton?.disabled) return;
        if (!window.confirm('현재 삽화 처리 흐름을 중단하시겠습니까?')) return;
        stopButton.disabled = true;
        stopButton.textContent = '중단 중…';
        try {
            const response = await fetch('/api/illustration_flow/cancel', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({run: runId}),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload.success) throw Error(payload.error || `중단 요청 실패 (${response.status})`);
            if (payload.flow && payload.flow.id === runId) receive(payload.flow, false);
            else await refresh();
            showToast('삽화 처리 흐름 중단을 요청했습니다.', 'success');
        } catch (error) {
            console.error('[ILLUST_FLOW] 중단 요청 실패:', error);
            showToast(error.message || '삽화 처리 흐름 중단에 실패했습니다.', 'error');
        } finally {
            updateStopButton();
        }
    }
    function updateZoomDisplay() {
        const resetZoom = modal?.querySelector('#if-zoom-reset');
        if (!resetZoom) return;
        const percentage = `${Math.round(scale * 100)}%`;
        resetZoom.textContent = percentage;
        resetZoom.setAttribute('aria-label', `현재 확대 비율 ${percentage}. 클릭하면 100%로 되돌립니다.`);
    }
    function zoom(delta) {scale = Math.max(0.4, Math.min(1.75, scale + delta)); updateZoomDisplay(); render();}
    let toastHome;
    function rehomeToast() {
        const toast = document.getElementById('toast');
        if (!toast) return;
        toastHome ||= toast.parentElement;
        const host = developerModal?.open
            ? developerModal
            : detailModal?.open
                ? detailModal
                : modal?.open
                    ? modal
                    : toastHome;
        if (toast.parentElement !== host) host.append(toast);
    }
    function hideTooltip() {modal?.querySelector('.if-tooltip')?.remove();}
    function tooltip(n, port) {
        let tip = modal.querySelector('.if-tooltip');
        if (tip?.dataset.nodeId !== n.id) {hideTooltip(); tip = null;}
        const executor = nodeExecutor(n);
        const isOverallQualityReview = n.task_key === 'illustration_quality_inspection' && n.inspection_scope === 'overall';
        const action = isOverallQualityReview
            ? '클릭하면 전체 이미지를 한 화면에서 보고 일괄 평가할 수 있습니다.'
            : n.error || n.summary || '클릭하여 상세 보기';
        const timing = Object.entries(timingDetails(n)).map(([key, value]) => `${key}: ${value}`).join('\n');
        if (!tip) {
            tip = element('div', 'if-tooltip'); tip.dataset.nodeId = n.id;
            tip.setAttribute('role', 'tooltip'); modal.append(tip);
        }
        const model = n.mask_generation_mode === 'inline'
            ? `자체 생성 · ${n.source_call_name || '현재 모델'}${n.model ? ' · ' + n.model : ''} · 별도 LLM 호출 없음`
            : executor === 'llm' ? (window.llmRouteSlotModelName?.(n.llm_slot, true, n.model || '') || n.model) : n.model;
        updateText(tip, `${n.label}\n${executorLabels[executor] || executor} · ${nodeState(n)}${executor === 'llm' && Number.isInteger(n.attempt_count) ? ` · 시도 ${n.attempt_count}회` : ''}\n${model || (executor === 'llm' ? '모델 배정 대기' : executor === 'comfy' ? 'ComfyUI' : executor === 'human' ? '사람이 선택한 입력 상태' : '서버 처리')}\n${timing ? timing + '\n수신 시점 기준 · 최종 시도\n' : ''}${n.phase === 'fallback' ? '폴백 사용 · ' : ''}${action}`);
        const r = port.getBoundingClientRect();
        tip.style.left = `${Math.max(8, Math.min(innerWidth - tip.offsetWidth - 12, r.left - 80))}px`;
        tip.style.top = `${Math.max(8, Math.min(innerHeight - tip.offsetHeight - 12, r.bottom + 10))}px`;
    }
    function layoutNodes(nodes, kind) {
        // Display stages are fixed even when branches finish or dispatch images early.
        // Names also classify reserved LLM nodes before routing metadata arrives.
        const stageRules = [
            ['SLOT-SELECTER', 'call2_select', 2, 3],
            ['DETAIL', 'call2_detail', 4, 0],
            ['KEYVIS', 'call2_keyvis', 4, 1],
            ['POST PROCESS', 'call3', 4, 3],
            ['CHARACTER-RESOLVE', 'character_resolve', 1, 0],
            ['PROFILE-CONTEXT-CACHE', 'profile_context_translation', 1, 1],
            ['PROFILE-CONTEXT-TRANSLATE', 'profile_context_translation', 1, 1],
            ['PROFILE-RESOLVE', 'profile_resolve', 1, 2],
            ['OUTFIT', 'call1', 2, 1],
            ['CALL1', 'call1', 2, 1],
            ['TIME & PLACE', 'time_place', 2, 2],
            ['CALL2-SELECT', 'call2_select', 2, 3],
            ['CALL2-PLAN', 'call2_plan', 3, 0],
            ['CALL2-DETAIL', 'call2_detail', 4, 0],
            ['CALL2-KEYVIS', 'call2_keyvis', 4, 1],
            ['CALL2-FALLBACK', 'call2', 4, 0],
            ['CALL2', 'call2', 4, 0],
            ['CALL3', 'call3', 4, 3],
            ['MULTI-CHAR-MASK', 'illustration_multi_char_mask', 5, 0],
            ['MULTI-CHAR_MASK', 'illustration_multi_char_mask', 5, 0],
            ['ORIGINAL-ASSET-RECOVERY', 'original_asset_recovery', 5, 2],
            ['ORIGINAL-RECOVERY', 'original_asset_recovery', 5, 2],
            ['ORIGINAL-ASSET', 'original_asset', 5, 1],
            ['SLOT', 'illustration_images', 6, 0],
            ['이미지 생성', 'illustration_generation', 6, 1],
        ];
        const stageOf = n => {
            if (kind !== 'illustration' || n.kind === 'request' || n.kind === 'result') return null;
            for (const value of [n.call_name, n.label]) {
                const name = String(value || '');
                const rule = stageRules.find(([label]) => name === label || name.startsWith(`${label} `) || name.startsWith(`${label}-`));
                if (rule) return rule;
            }
            const group = String(n.layout_group || '');
            const rule = stageRules.find(([, groupName]) => groupName === group || `illustration_${groupName}` === n.task_key || groupName === n.task_key);
            return rule || (n.kind === 'image' ? stageRules.find(([, groupName]) => groupName === 'illustration_images') : null);
        };
        const stageById = new Map(nodes.map(n => [n.id, stageOf(n)]));
        const layoutGroup = n => stageById.get(n.id)
            ? `__illustration_step_${stageById.get(n.id)[2]}__`
            : kind === 'illustration' && n.kind !== 'request'
                ? String(n.layout_group || (n.task_key && `${n.task_key}:${n.inspection_scope || ''}`)
                    || `__followup_${n.kind || ''}_${n.call_name || n.label || n.id}__`)
                : String(n.layout_group || n.id);
        const nodeById = new Map(nodes.map(n => [n.id, n]));
        const membersByGroup = new Map(), dependenciesByGroup = new Map();
        nodes.forEach(n => {
            const group = layoutGroup(n);
            const members = membersByGroup.get(group) || [];
            members.push(n);
            membersByGroup.set(group, members);
            const dependencies = dependenciesByGroup.get(group) || new Set();
            (n.dependencies || []).forEach(parentId => {
                const parent = nodeById.get(parentId);
                if (!parent) return;
                const parentGroup = layoutGroup(parent);
                if (parentGroup !== group) dependencies.add(parentGroup);
            });
            dependenciesByGroup.set(group, dependencies);
        });
        const depthByGroup = new Map(), resolvingGroups = new Set();
        const groupDepth = group => {
            if (depthByGroup.has(group)) return depthByGroup.get(group);
            const members = membersByGroup.get(group) || [];
            const stage = members.map(n => stageById.get(n.id)).find(Boolean);
            if (stage) {depthByGroup.set(group, stage[2]); return stage[2];}
            if (members.some(n => n.kind === 'request')) {depthByGroup.set(group, 0); return 0;}
            if (resolvingGroups.has(group)) {
                console.error('[ILLUST_FLOW] layout_group dependency cycle', group);
                return kind === 'illustration' ? 7 : 1;
            }
            resolvingGroups.add(group);
            const dependencies = [...(dependenciesByGroup.get(group) || [])];
            const minimumDepth = kind === 'illustration' ? 7 : dependencies.length ? 1 : 0;
            const depth = Math.max(minimumDepth, ...dependencies.map(parentGroup => groupDepth(parentGroup) + 1));
            resolvingGroups.delete(group);
            depthByGroup.set(group, depth);
            return depth;
        };
        const insertionOrder = new Map(nodes.map((n, index) => [n.id, index]));
        const orderedNodes = [...nodes].sort((a, b) => {
            const stageA = stageById.get(a.id), stageB = stageById.get(b.id);
            const stageOrder = (stageA?.[3] || 0) - (stageB?.[3] || 0);
            const orderA = Number.isFinite(Number(a.layout_order)) ? Number(a.layout_order) : 0;
            const orderB = Number.isFinite(Number(b.layout_order)) ? Number(b.layout_order) : 0;
            const nameOrder = kind === 'illustration' || (stageA && stageB)
                ? String(a.call_name || a.label || '').localeCompare(String(b.call_name || b.label || ''), undefined, {numeric: true})
                : 0;
            const tieOrder = kind === 'illustration' || (stageA && stageB)
                ? String(a.id).localeCompare(String(b.id), undefined, {numeric: true})
                : insertionOrder.get(a.id) - insertionOrder.get(b.id);
            return stageOrder || orderA - orderB || nameOrder || tieOrder;
        });
        const positions = new Map(), layers = new Map(), rootColumnX = 36;
        const top = kind === 'illustration' ? 104 : 32;
        const depthById = new Map(nodes.map(n => [n.id, groupDepth(layoutGroup(n))]));
        const laneById = new Map();
        if (kind === 'illustration') {
            const plans = orderedNodes.filter(n => stageById.get(n.id)?.[1] === 'call2_plan');
            const detailsByPlan = new Map(plans.map(n => [n.id, []]));
            const planAncestors = (id, visiting = new Set()) => {
                if (!nodeById.has(id) || visiting.has(id)) return new Set();
                if (detailsByPlan.has(id)) return new Set([id]);
                const nextVisiting = new Set(visiting); nextVisiting.add(id);
                return new Set((nodeById.get(id).dependencies || [])
                    .flatMap(parent => [...planAncestors(parent, nextVisiting)]));
            };
            orderedNodes.filter(n => stageById.get(n.id)?.[1] === 'call2_detail').forEach(n => {
                const parents = planAncestors(n.id);
                // Aggregate DETAIL calls have no single PLAN row to align with.
                if (parents.size === 1) detailsByPlan.get([...parents][0]).push(n);
            });
            if ([...detailsByPlan.values()].some(details => details.length)) {
                let nextLane = 0;
                plans.forEach(plan => {
                    const details = detailsByPlan.get(plan.id);
                    laneById.set(plan.id, nextLane);
                    details.forEach((detail, index) => laneById.set(detail.id, nextLane + index));
                    // Keep later PLAN rows reserved while their DETAIL calls arrive.
                    nextLane += Math.max(1, details.length);
                });
                layers.set(3, nextLane); layers.set(4, nextLane);
            }
            const branchNodes = orderedNodes.filter(n => depthById.get(n.id) >= 5 && n.kind !== 'result');
            const branchIds = new Set(branchNodes.map(n => n.id));
            const keyById = new Map(), keyByPrompt = new Map(), children = new Map();
            branchNodes.forEach(n => {
                // Parse declared slot identifiers in task captions, never prompt/context prose.
                const slot = illustrationSlot(n);
                if (slot !== null) {
                    const key = `slot:${slot}`;
                    keyById.set(n.id, key);
                    if (n.prompt_id) keyByPrompt.set(n.prompt_id, key);
                }
                (n.dependencies || []).forEach(id => {
                    if (!branchIds.has(id)) return;
                    const members = children.get(id) || [];
                    members.push(n.id); children.set(id, members);
                });
            });
            branchNodes.forEach(n => {
                if (!keyById.has(n.id) && n.prompt_id) keyById.set(n.id, keyByPrompt.get(n.prompt_id) || `prompt:${n.prompt_id}`);
                if (!keyById.has(n.id) && n.kind === 'image') keyById.set(n.id, `image:${n.id}`);
            });
            const relatedKeys = (id, downstream, visiting = new Set()) => {
                if (!branchIds.has(id) || visiting.has(id)) return new Set();
                if (keyById.has(id)) return new Set([keyById.get(id)]);
                const nextVisiting = new Set(visiting); nextVisiting.add(id);
                const related = downstream ? children.get(id) || [] : nodeById.get(id).dependencies || [];
                return new Set(related.flatMap(next => [...relatedKeys(next, downstream, nextVisiting)]));
            };
            const inferredKeys = new Map();
            branchNodes.forEach(n => {
                if (keyById.has(n.id)) return;
                const ancestors = relatedKeys(n.id, false);
                const keys = ancestors.size ? ancestors : relatedKeys(n.id, true);
                // Shared/aggregate work must not be attached to an arbitrary image.
                if (keys.size === 1) inferredKeys.set(n.id, [...keys][0]);
            });
            inferredKeys.forEach((key, id) => keyById.set(id, key));
            const keys = [...new Set(keyById.values())].sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
            const counts = new Map();
            branchNodes.forEach(n => {
                const key = keyById.get(n.id);
                if (!key) return;
                const byDepth = counts.get(key) || new Map(), depth = depthById.get(n.id);
                byDepth.set(depth, (byDepth.get(depth) || 0) + 1); counts.set(key, byDepth);
            });
            let nextLane = 0;
            const offsets = new Map();
            keys.forEach(key => {
                offsets.set(key, nextLane);
                // Reserve room for retries or multiple same-stage tasks without overlap.
                nextLane += Math.max(...counts.get(key).values());
            });
            const used = new Map();
            branchNodes.forEach(n => {
                const key = keyById.get(n.id);
                if (!key) return;
                const cell = `${depthById.get(n.id)}:${key}`, index = used.get(cell) || 0;
                used.set(cell, index + 1); laneById.set(n.id, offsets.get(key) + index);
            });
            if (nextLane) new Set([...depthById.values()].filter(depth => depth >= 5)).forEach(depth => layers.set(depth, nextLane));
        }
        // PLAN/DETAIL share parent rows; later stages share rows by illustration.
        orderedNodes.forEach(n => {
            const depth = depthById.get(n.id);
            const lane = laneById.has(n.id) ? laneById.get(n.id) : layers.get(depth) || 0;
            if (!laneById.has(n.id)) layers.set(depth, lane + 1);
            positions.set(n.id, {depth, x: rootColumnX + depth * 290, y: lane});
        });
        const rowHeights = new Map();
        nodes.forEach(n => {
            const p = positions.get(n.id), group = kind === 'illustration' && p.depth >= 5 ? 'branches' : p.depth;
            const key = `${group}:${p.y}`;
            const cardHeight = kind === 'illustration' && nodePreviewKind(n) ? 360 : 98;
            rowHeights.set(key, Math.max(rowHeights.get(key) || 0, cardHeight));
        });
        positions.forEach(p => {
            const group = kind === 'illustration' && p.depth >= 5 ? 'branches' : p.depth;
            let y = top;
            for (let row = 0; row < p.y; row += 1) y += (rowHeights.get(`${group}:${row}`) || 98) + 38;
            p.y = y;
        });
        const width = Math.max(kind === 'illustration' ? rootColumnX + 7 * 290 + 270 : 650, ...[...positions.values()].map(p => p.x + 270));
        const height = Math.max(350, ...nodes.map(n => positions.get(n.id).y + (kind === 'illustration' && nodePreviewKind(n) ? 360 : 98) + 42));
        return {positions, width, height};
    }
    function nodePreviewKind(n) {
        const caption = String(n.call_name || n.label || '');
        if (n.task_key === 'illustration_multi_char_mask' || n.layout_group === 'illustration_multi_char_mask'
                || caption === 'MULTI-CHAR-MASK' || caption.startsWith('MULTI-CHAR-MASK ')
                || caption === 'MULTI-CHAR_MASK' || caption.startsWith('MULTI-CHAR_MASK ')) return 'mask';
        return n.kind === 'image' || n.layout_group === 'illustration_generation' ? 'image' : '';
    }
    function nodeImageUrl(n, runStatus) {
        if (n.image_url) return n.image_url;
        return n.kind === 'image' && n.status === 'completed' && runStatus === 'completed' && n.prompt_id
            ? `/view?filename=${encodeURIComponent(`ComfyUI_${String(n.prompt_id).slice(0, 8)}.png`)}` : '';
    }
    function illustrationSlot(n) {
        const caption = String(n.call_name || n.label || '');
        const slot = n.illustration_slot ?? (n.inspection_images?.length === 1 ? n.inspection_images[0].slot : undefined)
            ?? caption.match(/\bslot(?:\s*=\s*|\s+)(-?\d+)\b/)?.[1];
        return slot !== undefined && slot !== null && slot !== '' && Number.isInteger(Number(slot)) ? Number(slot) : null;
    }
    function imageMaskNode(n, flow) {
        const masks = (flow.nodes || []).filter(node => nodePreviewKind(node) === 'mask');
        const slot = illustrationSlot(n);
        const matching = slot === null ? [] : masks.filter(mask => illustrationSlot(mask) === slot);
        const dependencies = masks.filter(mask => n.dependencies?.includes(mask.id));
        const candidates = matching.length ? matching
            : dependencies.length === 1 && (slot === null || illustrationSlot(dependencies[0]) === null) ? dependencies : [];
        return candidates.findLast(mask => mask.status === 'completed') || candidates.at(-1);
    }
    function imageMaskOverlay(layout, img, maskUrl) {
        const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
        const width = img.naturalWidth || layout?.mask_width || 100, height = img.naturalHeight || layout?.mask_height || 100;
        svg.classList.add('if-image-mask-overlay'); svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        svg.setAttribute('aria-hidden', 'true');
        if (maskUrl) {
            const mask = document.createElementNS(ns, 'image');
            mask.setAttribute('href', maskUrl); mask.setAttribute('width', width); mask.setAttribute('height', height);
            mask.setAttribute('preserveAspectRatio', 'none');
            mask.onerror = () => console.error('[ILLUST_FLOW] 생성 마스크 표시 실패:', {url: maskUrl});
            svg.style.mixBlendMode = 'screen'; svg.style.opacity = '.35';
            svg.append(mask);
            return svg;
        }
        const colors = ['#ef4444', '#22c55e', '#3b82f6'];
        [...layout.regions].sort((a, b) => a.x + a.width / 2 - b.x - b.width / 2).forEach((region, index) => {
            const rect = document.createElementNS(ns, 'rect'), color = colors[index % colors.length];
            for (const field of ['x', 'y', 'width', 'height']) rect.setAttribute(field, Number(region[field]) * (field === 'x' || field === 'width' ? width : height));
            rect.setAttribute('fill', color); rect.setAttribute('fill-opacity', '.28');
            rect.setAttribute('stroke', color); rect.setAttribute('stroke-width', '1.5'); rect.setAttribute('vector-effect', 'non-scaling-stroke');
            svg.append(rect);
        });
        return svg;
    }
    function maskResultKey(flow, n) {return `${flow.id}:${n.id}:${n.ended_at || n.response_ended_at || ''}`;}
    async function loadMaskResults(flow, nodes) {
        if (maskResultsRun !== flow.id) {maskResults.clear(); maskResultsRun = flow.id;}
        await Promise.all(nodes.filter(n => nodePreviewKind(n) === 'mask' && n.status === 'completed').map(async n => {
            const key = maskResultKey(flow, n);
            if (maskResults.has(key)) return;
            maskResults.set(key, {pending: true});
            try {
                const response = await fetch(`/api/illustration_flow?run=${encodeURIComponent(flow.id)}&node=${encodeURIComponent(n.id)}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const {node} = await response.json();
                const raw = detailOutput(node || {});
                const text = typeof raw === 'string' ? raw : JSON.stringify(raw);
                const layout = typeof raw === 'object' ? raw : JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1));
                if (!Array.isArray(layout?.regions)) throw new Error('영역 결과가 없습니다');
                if (maskResultsRun !== flow.id) return;
                maskResults.set(key, {layout});
            } catch (error) {
                console.error('[ILLUST_FLOW] 노드 영역 결과 표시 실패:', {run: flow.id, node: n.id}, error);
                if (maskResultsRun === flow.id) maskResults.set(key, {error: '영역 결과를 표시하지 못했습니다. 출력 상세를 확인하세요.'});
            }
        }));
        if (activeTab === 'illustration' && flows.illustration?.id === flow.id) render();
    }
    function nodePreview(n, flow) {
        const kind = nodePreviewKind(n), preview = element('div', 'if-node-preview');
        preview.append(element('div', 'if-preview-title', kind === 'mask' ? '생성 영역' : '생성 이미지'));
        if (kind === 'mask') {
            const result = maskResults.get(maskResultKey(flow, n));
            if (result?.layout) {
                const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
                const imageNode = (flow.nodes || []).find(child => child.kind === 'image' && child.dependencies?.includes(n.id));
                const image = imageNode && previewImages.get(imageNode.id);
                const canvasWidth = image?.naturalWidth || result.layout.mask_width || 100;
                const canvasHeight = image?.naturalHeight || result.layout.mask_height || 100;
                svg.classList.add('if-mask-preview'); svg.setAttribute('viewBox', `0 0 ${canvasWidth} ${canvasHeight}`);
                svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${n.label} 영역 배치`);
                const colors = ['#ef4444', '#22c55e', '#3b82f6'];
                const regions = [...result.layout.regions].sort((a, b) => a.x + a.width / 2 - b.x - b.width / 2);
                regions.forEach((r, index) => {
                    const rect = document.createElementNS(ns, 'rect');
                    for (const field of ['x', 'y', 'width', 'height']) rect.setAttribute(field, Number(r[field]) * (field === 'x' || field === 'width' ? canvasWidth : canvasHeight));
                    rect.setAttribute('fill', colors[index % colors.length]); rect.setAttribute('fill-opacity', '.65');
                    rect.setAttribute('stroke', colors[index % colors.length]); rect.setAttribute('stroke-width', '.7');
                    const title = document.createElementNS(ns, 'title'); title.textContent = r.name; rect.append(title); svg.append(rect);
                });
                preview.append(svg);
                regions.forEach((r, index) => {
                    const percent = v => `${(Number(v) * 100).toFixed(0)}%`;
                    const line = element('div', 'if-mask-region', `${'RGB'[index]} · ${r.name} · 면적 ${(r.width * r.height * 100).toFixed(1)}%\nx ${percent(r.x)}, y ${percent(r.y)} · 폭 ${percent(r.width)}, 높이 ${percent(r.height)}`);
                    line.style.color = ['#fecaca', '#bbf7d0', '#bfdbfe'][index % colors.length]; line.style.whiteSpace = 'pre-line'; preview.append(line);
                });
                preview.append(element('pre', 'if-mask-text', `구도: ${result.layout.composition_prompt || ''}\n배경: ${result.layout.background_prompt || ''}`));
            } else preview.append(element('div', 'if-preview-empty', result?.error || (n.status === 'completed' ? '영역 결과 불러오는 중…' : n.error || '영역 생성 대기')));
        } else {
            const url = nodeImageUrl(n, flow.status);
            if (url) {
                const imageButton = button('', () => openDetail(n.id)); imageButton.className = 'if-preview-image-button';
                imageButton.setAttribute('aria-label', `${n.label} 생성 이미지 크게 보기`);
                const img = previewImages.get(n.id) || element('img', 'if-preview-image');
                if (img.getAttribute('src') !== url) {img.src = url; delete img.dataset.failed;}
                img.alt = `${n.label} 생성 결과`; img.loading = 'lazy'; previewImages.set(n.id, img);
                img.onload = () => {if (activeTab === 'illustration' && flows.illustration?.id === flow.id) render();};
                img.onerror = () => {console.error('[ILLUST_FLOW] 생성 이미지 표시 실패:', {node: n.id, url}); img.dataset.failed = 'true'; imageButton.replaceChildren(element('div', 'if-preview-empty', '이미지를 불러오지 못했습니다.'));};
                const maskNode = imageMaskNode(n, flow);
                const result = maskNode && maskResults.get(maskResultKey(flow, maskNode));
                const recordedMask = Object.hasOwn(n, 'mask_url');
                if ((n.mask_url || (!recordedMask && maskNode)) && img.dataset.failed !== 'true') {
                    const toggle = element('label', 'if-mask-overlay-toggle'), input = element('input');
                    const overlay = n.mask_url ? imageMaskOverlay(null, img, n.mask_url)
                        : result?.layout ? imageMaskOverlay(result.layout, img) : null;
                    input.type = 'checkbox'; input.checked = maskOverlayNodes.has(n.id); input.disabled = !overlay;
                    input.dataset.maskOverlayNode = n.id; input.setAttribute('aria-label', `${n.label} 마스크 겹쳐 보기`);
                    toggle.title = result?.error || (overlay ? '생성 이미지 위에 인물 영역 마스크를 겹쳐 표시합니다.' : '마스크 결과를 기다리는 중입니다.');
                    if (overlay) {
                        overlay.toggleAttribute('hidden', !input.checked);
                        input.onchange = () => {
                            if (input.checked) maskOverlayNodes.add(n.id); else maskOverlayNodes.delete(n.id);
                            overlay.toggleAttribute('hidden', !input.checked);
                        };
                    }
                    toggle.append(input, document.createTextNode('마스크 겹쳐 보기')); preview.append(toggle);
                    imageButton.append(img);
                    if (overlay) imageButton.append(overlay);
                } else imageButton.append(img.dataset.failed ? element('div', 'if-preview-empty', '이미지를 불러오지 못했습니다.') : img);
                preview.append(imageButton);
            } else preview.append(element('div', 'if-preview-empty', n.error || '이미지 생성 대기'));
        }
        return preview;
    }
    function previewSignature(n, flow) {
        const kind = nodePreviewKind(n), mask = kind === 'image' ? imageMaskNode(n, flow) : n;
        const result = mask && maskResults.get(maskResultKey(flow, mask));
        const imageNode = kind === 'mask'
            ? (flow.nodes || []).find(child => child.kind === 'image' && child.dependencies?.includes(n.id)) : n;
        const img = imageNode && previewImages.get(imageNode.id);
        return JSON.stringify([kind, n.label, n.status, n.error, nodeImageUrl(n, flow.status),
            n.mask_url, Object.hasOwn(n, 'mask_url'), mask?.id, result,
            img?.naturalWidth, img?.naturalHeight, img?.dataset.failed]);
    }
    function render() {
        updateStopButton();
        if (!modal?.open) return;
        const viewport = modal.querySelector('.if-viewport');
        modal.querySelectorAll('.if-tab').forEach(tab => {
            const active = tab.dataset.tab === activeTab;
            tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1;
        });
        modal.querySelectorAll('.if-auto-open-setting').forEach(setting => {
            setting.hidden = setting.dataset.tab !== activeTab;
        });
        viewport.setAttribute('aria-labelledby', `if-tab-${activeTab}`);
        const scroll = [viewport.scrollLeft, viewport.scrollTop];
        const focusedNode = document.activeElement?.dataset?.nodeId;
        const focusedMaskOverlay = document.activeElement?.dataset?.maskOverlayNode;
        if (!flow) {
            hideTooltip();
            modal.querySelector('.if-subtitle').textContent = `${tabLabels[activeTab]} · 최신 요청의 실행 상태`;
            const name = activeTab === 'video_input' ? '영상 입력 개선' : activeTab === 'video' ? '영상' : '삽화';
            const empty = element('div', 'if-empty', `아직 ${name} 요청이 없습니다. 요청이 들어오면 여기에 처리 흐름이 표시됩니다.`);
            if (activeTab === 'video') empty.append(element('small', '', '연출 작성·참조 분석 → 프롬프트 작성·후보 선택 → 영상 생성 → 후처리 → 결과 반환'));
            if (activeTab === 'video_input') empty.append(element('small', '', '입력 상태와 AI 다듬기 결과, 적용·되돌리기 선택을 한 세션 안에 이어서 보존합니다.'));
            empty.append(element('small', '', `현재 확대 비율 ${Math.round(scale * 100)}% — 지금 조절한 비율은 다음 그래프에도 적용됩니다.`));
            viewport.replaceChildren(empty);
            return;
        }
        modal.querySelector('.if-subtitle').textContent = `${flow.label} · ${labels[flow.status] || flow.status} · ${new Date(flow.created_at * 1000).toLocaleString()}`;
        const nodes = flow.nodes || [];
        if (activeTab === 'illustration' && maskResultsRun !== flow.id) {
            maskResults.clear(); previewImages.clear(); maskOverlayNodes.clear(); maskResultsRun = flow.id;
        }
        const {positions, width, height} = layoutNodes(nodes, activeTab);
        let space = viewport.querySelector('.if-space'), canvas = space?.querySelector('.if-canvas');
        const sameGraph = canvas && space.dataset.runId === flow.id && space.dataset.tab === activeTab;
        // Streaming updates must keep the pointer-down target connected. Reuse
        // the graph and keyed children even when new branches change its layout.
        if (!sameGraph) {
            hideTooltip();
            space = element('div', 'if-space'); canvas = element('div', 'if-canvas');
            space.dataset.runId = flow.id; space.dataset.tab = activeTab;
            space.append(canvas); viewport.replaceChildren(space);
        }
        space.style.width = `${width * scale}px`; space.style.height = `${height * scale}px`;
        canvas.style.width = `${width}px`; canvas.style.height = `${height}px`; canvas.style.transform = `scale(${scale})`;
        const ns = 'http://www.w3.org/2000/svg';
        let svg = canvas.querySelector('.if-edges');
        if (!svg) {
            svg = document.createElementNS(ns, 'svg'); svg.classList.add('if-edges');
            svg.setAttribute('aria-hidden', 'true'); canvas.append(svg);
        }
        svg.setAttribute('width', width); svg.setAttribute('height', height);
        if (activeTab === 'illustration') {
            if (!canvas.querySelector('.if-direction')) canvas.insertBefore(element('div', 'if-direction', '실행 순서 →'), svg);
            const columns = new Set([0, 1, 2, 3, 4, 5, 6, ...[...positions.values()].map(p => p.depth)]);
            const existingColumns = new Map([...canvas.querySelectorAll('.if-step-column')].map(column => [Number(column.dataset.step), column]));
            [...columns].sort((a, b) => a - b).forEach(depth => {
                let column = existingColumns.get(depth);
                if (!column) {
                    column = element('section', 'if-step-column'); column.dataset.step = String(depth);
                    column.append(element('div', 'if-step-title', depth === 0 ? '요청' : depth <= 6 ? `STEP ${depth}` : '후속 처리'));
                    canvas.insertBefore(column, svg);
                }
                column.style.left = `${24 + depth * 290}px`; column.style.height = `${height - 56}px`;
            });
            existingColumns.forEach((column, depth) => {if (!columns.has(depth)) column.remove();});
        }
        const paths = new Map([...svg.children].map(path => [path.dataset.edgeKey, path]));
        const edgeKeys = new Set();
        nodes.forEach(n => (n.dependencies || []).forEach(id => {
            const a = positions.get(id), b = positions.get(n.id); if (!a || !b) return;
            const key = JSON.stringify([id, n.id]); edgeKeys.add(key);
            let path = paths.get(key);
            if (!path) {path = document.createElementNS(ns, 'path'); path.dataset.edgeKey = key; svg.append(path);}
            path.setAttribute('d', a.depth === b.depth
                ? `M ${a.x + 222} ${a.y + 48} C ${a.x + 257} ${a.y + 48}, ${b.x + 257} ${b.y + 48}, ${b.x + 222} ${b.y + 48}`
                : `M ${a.x + 222} ${a.y + 48} C ${a.x + 257} ${a.y + 48}, ${b.x - 35} ${b.y + 48}, ${b.x} ${b.y + 48}`);
            path.setAttribute('fill', 'none'); path.setAttribute('stroke', colors[n.status] || '#64748b'); path.setAttribute('stroke-width', '1.8'); path.setAttribute('opacity', '.65');
            if (n.status === 'waiting') path.setAttribute('stroke-dasharray', '5 5');
            else path.removeAttribute('stroke-dasharray');
        }));
        paths.forEach((path, key) => {if (!edgeKeys.has(key)) path.remove();});
        const cards = new Map([...canvas.querySelectorAll('.if-node')].map(card => [card.dataset.nodeId, card]));
        const nodeIds = new Set(nodes.map(n => n.id));
        nodes.forEach(n => {
            const p = positions.get(n.id), executor = nodeExecutor(n);
            let card = cards.get(n.id);
            if (!card) {
                card = element('article', 'if-node'); card.dataset.nodeId = n.id;
                const heading = element('div', 'if-node-heading');
                heading.append(element('div', 'if-node-title'), element('span', 'if-node-attempts'));
                card.append(heading, element('div', 'if-node-state'), element('div', 'if-node-model'));
                const port = button('', () => openDetail(n.id)); port.className = 'if-port'; port.dataset.nodeId = n.id;
                card.append(port); canvas.append(card);
            }
            const isQualityReview = n.task_key === 'illustration_quality_inspection' && n.inspection_scope === 'overall';
            card.dataset.status = n.status; card.dataset.nodeId = n.id; card.dataset.executor = executor; card.dataset.reviewNode = String(isQualityReview); card.style.setProperty('--state', colors[n.status] || colors.waiting); card.style.setProperty('--node-tint', executorColors[executor] || executorColors.process);
            card.dataset.step = String(p.depth);
            card.style.left = `${p.x}px`; card.style.top = `${p.y}px`;
            const previewKind = activeTab === 'illustration' && nodePreviewKind(n);
            let preview = card.querySelector('.if-node-preview');
            if (previewKind) {
                card.style.height = '360px'; card.dataset.previewKind = previewKind;
                const signature = previewSignature(n, flow);
                if (!preview || preview.dataset.signature !== signature) {
                    const previewScroll = [preview?.scrollLeft || 0, preview?.scrollTop || 0];
                    const nextPreview = nodePreview(n, flow); nextPreview.dataset.signature = previewSignature(n, flow);
                    if (preview) preview.replaceWith(nextPreview); else card.append(nextPreview);
                    [nextPreview.scrollLeft, nextPreview.scrollTop] = previewScroll;
                }
            } else if (preview) {
                preview.remove(); card.style.height = ''; delete card.dataset.previewKind;
            }
            let title = card.querySelector('.if-node-title');
            const hasStateTable = ['OUTFIT', 'CALL1', 'TIME & PLACE'].includes(n.label);
            if (title.tagName !== (hasStateTable ? 'BUTTON' : 'DIV')) {
                const nextTitle = hasStateTable ? button(n.label, () => openDetail(n.id)) : element('div');
                title.replaceWith(nextTitle); title = nextTitle;
            }
            updateText(title, n.label);
            title.className = `if-node-title${hasStateTable ? ' if-call1-open' : ''}`; title.title = n.label;
            const attempts = card.querySelector('.if-node-attempts');
            attempts.hidden = executor !== 'llm' || !Number.isInteger(n.attempt_count);
            updateText(attempts, attempts.hidden ? '' : `시도 ${n.attempt_count}회`);
            attempts.title = '최초 호출·재시도·폴백을 포함한 총 시도 횟수';
            const modelText = isQualityReview
                ? '출력 ● 클릭 → 이미지 일괄 평가'
                : n.mask_generation_mode === 'inline' ? `자체 생성 · ${n.source_call_name || '현재 모델'}`
                : (executor === 'llm' ? (window.llmRouteSlotModelName?.(n.llm_slot, true, n.model || '') || n.model) : n.model) || (executor === 'llm' ? 'LLM' : executor === 'comfy' ? 'ComfyUI' : executor === 'human' ? '사람이 선택한 입력 상태' : '서버 처리');
            updateText(card.querySelector('.if-node-state'), nodeState(n));
            updateText(card.querySelector('.if-node-model'), modelText);
            let timing = card.querySelector('.if-node-timing');
            if (executor === 'llm') {
                if (!timing) {timing = element('div', 'if-node-timing'); card.append(timing);}
                updateText(timing, llmTiming(n));
            } else timing?.remove();
            const port = card.querySelector('.if-port');
            port.setAttribute('aria-label', isQualityReview ? `${n.label} 전체 이미지 보기 및 일괄 평가` : `${n.label} 출력 상세`);
            if (isQualityReview) port.title = '전체 생성 이미지를 한 화면에서 보고 평가합니다.';
            else port.removeAttribute('title');
            port.onmouseenter = port.onfocus = () => tooltip(n, port); port.onmouseleave = port.onblur = hideTooltip;
        });
        cards.forEach((card, id) => {if (!nodeIds.has(id)) {card.remove(); previewImages.delete(id); maskOverlayNodes.delete(id);}});
        viewport.scrollLeft = scroll[0]; viewport.scrollTop = scroll[1];
        if (sameGraph && focusedNode && document.activeElement?.dataset?.nodeId !== focusedNode) [...canvas.querySelectorAll('.if-port')].find(p => p.dataset.nodeId === focusedNode)?.focus({preventScroll: true});
        if (sameGraph && focusedMaskOverlay && document.activeElement?.dataset?.maskOverlayNode !== focusedMaskOverlay) [...canvas.querySelectorAll('.if-mask-overlay-toggle input')].find(input => input.dataset.maskOverlayNode === focusedMaskOverlay)?.focus({preventScroll: true});
        const tooltipNodeId = modal.querySelector('.if-tooltip')?.dataset.nodeId;
        if (tooltipNodeId) {
            const n = nodes.find(n => n.id === tooltipNodeId), port = [...canvas.querySelectorAll('.if-port')].find(p => p.dataset.nodeId === tooltipNodeId);
            if (n && port) tooltip(n, port); else hideTooltip();
        }
        if (activeTab === 'illustration' && nodes.some(n => nodePreviewKind(n) === 'mask' && n.status === 'completed' && !maskResults.has(maskResultKey(flow, n)))) void loadMaskResults(flow, nodes);
    }
    async function refresh(kind) {
        if (!kind) {await Promise.all(Object.keys(flows).map(key => refresh(key))); return;}
        const before = flows[kind];
        try {
            const response = await fetch(flowUrl(kind), {cache: 'no-store'});
            if (!response.ok) throw Error(`흐름 조회 실패 (${response.status})`);
            const next = (await response.json()).flow;
            if (!next && flows[kind] === before) {
                flows[kind] = null;
                if (kind === activeTab) {flow = null; if (detailModal?.open) detailModal.close(); render();}
            } else receive(next, false, kind);
        } catch (error) {console.error('[ILLUST_FLOW] 최신 상태 조회 실패:', error); if (modal?.open) showToast(error.message, 'error');}
    }
    function receive(next, autoOpen, kind = next?.kind || 'illustration') {
        if (!next) return;
        const current = flows[kind];
        if (current && (next.created_at < current.created_at || (next.id === current.id && next.revision <= current.revision))) return;
        const isNew = current?.id !== next.id;
        for (const node of next.nodes || []) {
            for (const field of ['label', 'source_call_name']) {
                if (typeof node[field] === 'string') node[field] = node[field].replaceAll('CALL2-SELECT', 'SLOT-SELECTER').replaceAll('CALL2-DETAIL', 'DETAIL').replaceAll('CALL2-KEYVIS', 'KEYVIS').replaceAll('CALL3', 'POST PROCESS');
            }
        }
        flows[kind] = next;
        if (autoOpen && autoOpenEnabled[kind] && isNew && !modal?.open) {
            init(); previousFocus = document.activeElement; switchTab(kind);
            flowBackdrop.hidden = false; modal.show(); modal.dispatchEvent(new Event('if-open'));
        }
        if (kind !== activeTab) return;
        flow = next;
        if (isNew && detailModal?.open) detailModal.close();
        render();
        if (selected && detailModal?.open) openDetail(selected, true);
    }
    function appendQualityReviewPanel(n, body, request, runId) {
        const panel = element('section', 'if-review-panel');
        panel.append(element('div', 'if-review-title', '전체 이미지 일괄 평가'));
        const guide = element('p', 'if-review-guide', '전체 검사에 포함된 이미지를 한 화면에서 확인한 뒤 각 이미지에 좋음·보통·나쁨과 리뷰를 입력하세요. 모든 평가를 선택한 뒤 한 번에 저장하면 연결된 LLM 흐름도 자동 정리에서 보호됩니다.');
        panel.append(guide);
        body.append(panel);
        if (!n.history_id) {
            panel.append(element('div', 'if-review-status', n.status === 'processing' || n.status === 'waiting'
                ? '전체 검사가 끝나면 이미지별 평가 입력란이 여기에 표시됩니다.'
                : '이 전체 검사에는 사람 평가를 연결할 기록 ID가 없습니다.'));
            return;
        }
        const reviewUrl = `/api/illustration_quality_inspection/review/${encodeURIComponent(n.history_id)}`;
        const loading = element('div', 'if-review-status', '전체 이미지와 저장된 평가를 불러오는 중…');
        panel.append(loading);
        fetch(reviewUrl, {cache: 'no-store'})
            .then(async response => {
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) throw Error(payload.error || `평가 정보 조회 실패 (${response.status})`);
                return payload;
            })
            .then(payload => {
                if (request !== detailRequest || runId !== flow?.id || !detailModal?.open || !panel.isConnected) return;
                panel.replaceChildren(element('div', 'if-review-title', '전체 이미지 일괄 평가'), guide);
                const images = Array.isArray(payload.images) ? payload.images : [];
                if (!images.length) {
                    panel.append(element('div', 'if-review-missing', '이 전체 검사 기록에는 연결된 이미지가 없습니다.'));
                    return;
                }
                const list = element('div', 'if-review-list');
                const entries = [];
                let saveButton = null;
                const status = element('span', 'if-review-status');
                const updateSaveState = () => {
                    const evaluable = entries.filter(entry => entry.historyId);
                    const selectedCount = evaluable.filter(entry => entry.rating).length;
                    if (saveButton) saveButton.disabled = !evaluable.length || selectedCount !== evaluable.length;
                    if (!evaluable.length) status.textContent = '평가를 저장할 수 있는 개별 검사 기록이 없습니다.';
                    else if (selectedCount !== evaluable.length) status.textContent = `평가 선택 ${selectedCount}/${evaluable.length} · 모두 선택하면 한 번에 저장할 수 있습니다.`;
                    else status.textContent = `평가 선택 완료 ${selectedCount}/${evaluable.length} · 전체 저장 준비됨`;
                };
                images.forEach((image, index) => {
                    const item = element('section', 'if-review-item');
                    const figure = element('figure', 'if-review-figure');
                    const slot = image.slot ?? index + 1;
                    if (image.image_url) {
                        const img = element('img', 'if-review-image');
                        img.src = image.image_url;
                        img.alt = `생성 이미지 slot ${slot}`;
                        img.loading = 'lazy';
                        img.addEventListener('error', () => {
                            console.error('[ILLUST_QUALITY] 일괄 평가 이미지 표시 실패:', image);
                            img.replaceWith(element('div', 'if-review-missing', '이미지 파일을 표시할 수 없습니다.'));
                        }, {once: true});
                        figure.append(img);
                    } else {
                        figure.append(element('div', 'if-review-missing', '이미지 파일이 삭제되었거나 이동되었습니다.'));
                    }
                    figure.append(element('figcaption', 'if-review-caption', `slot ${slot} · ${image.backup_name || '백업명 없음'}`));
                    item.append(figure);
                    const itemBody = element('div', 'if-review-item-body');
                    if (image.automatic_output) {
                        const automatic = element('details', 'if-review-auto');
                        automatic.append(element('summary', '', '자동 검사 결과'), element('pre', '', String(image.automatic_output)));
                        itemBody.append(automatic);
                    }
                    const historyId = String(image.review_history_id || '');
                    if (!historyId) {
                        itemBody.append(element('div', 'if-review-status', '이 이미지의 개별 검사 기록을 찾을 수 없어 사람 평가를 저장할 수 없습니다.'));
                        item.append(itemBody); list.append(item); return;
                    }
                    itemBody.append(element('div', '', '내 평가'));
                    const ratings = element('div', 'if-review-ratings');
                    const ratingButtons = [];
                    const entry = {historyId, rating: String(image.human_evaluation?.rating || ''), reason: null};
                    entries.push(entry);
                    const selectRating = rating => {
                        entry.rating = rating;
                        ratingButtons.forEach(ratingButton => ratingButton.setAttribute('aria-pressed', String(ratingButton.dataset.rating === rating)));
                        updateSaveState();
                    };
                    for (const [rating, label] of [['good', '좋음'], ['normal', '보통'], ['bad', '나쁨']]) {
                        const ratingButton = button(label, () => selectRating(rating));
                        ratingButton.classList.add('if-review-rating');
                        ratingButton.dataset.rating = rating;
                        ratingButton.setAttribute('aria-pressed', String(entry.rating === rating));
                        ratingButtons.push(ratingButton);
                        ratings.append(ratingButton);
                    }
                    itemBody.append(ratings);
                    const reason = element('textarea', 'if-review-reason');
                    reason.value = String(image.human_evaluation?.reason || '');
                    reason.placeholder = '예: 구도는 좋지만 표정과 접촉 표현이 어색함';
                    reason.setAttribute('aria-label', `slot ${slot} 평가 리뷰`);
                    entry.reason = reason;
                    itemBody.append(reason);
                    const updatedAt = String(image.human_evaluation?.updated_at || '');
                    if (updatedAt) itemBody.append(element('div', 'if-review-status', `기존 평가 저장 시각 · ${updatedAt}`));
                    item.append(itemBody);
                    list.append(item);
                });
                panel.append(list);
                const actions = element('div', 'if-review-actions');
                saveButton = button('전체 평가 저장', async () => {
                    const evaluable = entries.filter(entry => entry.historyId);
                    if (!evaluable.length || evaluable.some(entry => !entry.rating) || saveButton.disabled) return;
                    const reviews = evaluable.map(entry => ({history_id: entry.historyId, rating: entry.rating, reason: entry.reason?.value || ''}));
                    saveButton.disabled = true;
                    status.textContent = `${reviews.length}개 이미지의 평가와 연결된 LLM 흐름을 저장하는 중…`;
                    try {
                        const response = await fetch(reviewUrl, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({reviews}),
                        });
                        const saved = await response.json().catch(() => ({}));
                        if (!response.ok) throw Error(saved.error || `일괄 평가 저장 실패 (${response.status})`);
                        const retained = Number(saved.retained_history_count || 0);
                        const missing = Array.isArray(saved.missing_history_ids) ? saved.missing_history_ids.length : 0;
                        status.textContent = missing
                            ? `전체 평가 저장됨 · LLM 기록 ${retained}건 보존 · 이미 삭제된 ${missing}건은 복구할 수 없습니다.`
                            : `전체 평가 저장됨 · 이미지와 연결된 LLM 기록 ${retained}건을 자동 정리에서 보호합니다.`;
                        status.style.color = missing ? '#fde68a' : '#bbf7d0';
                        qualityInspectionToast(`${reviews.length}개 이미지 평가를 한 번에 저장했습니다.`, 'success');
                    } catch (error) {
                        console.error('[ILLUST_QUALITY] 이미지 일괄 평가 저장 실패:', {historyId: n.history_id, error});
                        status.textContent = `일괄 평가 저장 실패: ${error.message || error}`;
                        status.style.color = '#fecdd3';
                        qualityInspectionToast(`이미지 일괄 평가 저장 실패: ${error.message || error}`, 'error');
                    } finally {
                        saveButton.disabled = false;
                    }
                });
                saveButton.classList.add('if-review-save');
                actions.append(saveButton, status);
                panel.append(actions);
                updateSaveState();
            })
            .catch(error => {
                console.error('[ILLUST_QUALITY] 일괄 평가 정보 조회 실패:', {historyId: n.history_id, error});
                if (request !== detailRequest || runId !== flow?.id || !detailModal?.open || !panel.isConnected) return;
                loading.textContent = n.status === 'processing' || n.status === 'waiting'
                    ? '전체 검사가 끝나면 이미지별 평가 입력란이 여기에 표시됩니다.'
                    : `평가 정보를 불러오지 못했습니다: ${error.message || error}`;
                loading.style.color = n.status === 'processing' || n.status === 'waiting' ? '' : '#fecdd3';
            });
    }
    function appendTrackedStateTable(n, body) {
        const timePlace = !!n.time_place_state_table;
        const data = timePlace ? n.time_place_state_table : n.call1_state_table;
        if (!data) return;
        const title = timePlace ? 'TIME & PLACE 시간표' : '복장·헤어 시간표';
        const fields = timePlace ? [['time', 'TIME'], ['place', 'PLACE']] : [['outfit', '복장'], ['hair', '헤어']];
        body.append(element('h3', '', title), element('p', 'if-subtitle', `왼쪽에서 오른쪽으로 상태를 이어받습니다. 파란색은 ${timePlace ? 'TIME & PLACE' : 'OUTFIT'}가 직접 출력한 값이고, 회색은 앞 SLOT에서 이어받은 값입니다. 각 항목은 새 값이 나오는 지점부터 독립적으로 바뀝니다.`));
        const wrap = element('div', 'if-state-table-wrap'), table = element('table', 'if-state-table');
        wrap.dataset.detailScrollKey = timePlace ? 'time-place-state-table' : 'call1-state-table';
        table.setAttribute('aria-label', timePlace ? 'TIME & PLACE 슬롯별 시간·장소 상태' : 'OUTFIT 슬롯별 복장·헤어 상태');
        const head = element('thead'), header = element('tr'), corner = element('th', '', '인물'); corner.scope = 'col'; header.append(corner);
        data.slots.forEach(slot => {const th = element('th', '', `SLOT${slot}`); th.scope = 'col'; header.append(th);});
        head.append(header); table.append(head);
        const rows = element('tbody');
        data.characters.forEach(name => {
            const row = element('tr'), label = element('th', '', name); label.scope = 'row'; row.append(label);
            data.slots.forEach(slot => {
                const key = `SLOT${slot}`, cell = element('td');
                const state = data.states[key][name], changes = data.changes[key]?.[name] || {};
                fields.forEach(([field, caption]) => {
                    const direct = Object.hasOwn(changes, field), section = element('div', `if-state-tags ${direct ? 'if-state-direct' : 'if-state-inherited'}`);
                    section.dataset.field = field; section.dataset.source = direct ? 'direct' : 'inherited';
                    section.append(element('div', 'if-state-tag-label', `${caption} · ${direct ? '직접' : '이어받음'}`), element('div', '', state[field] || (timePlace ? '(정보 없음)' : '(태그 없음)'))); cell.append(section);
                });
                row.append(cell);
            });
            rows.append(row);
        });
        table.append(rows); wrap.append(table); body.append(wrap);
    }

    function detailText(text, key) {
        const pre = element('pre', '', text);
        pre.dataset.detailScrollKey = key;
        return pre;
    }

    function captureDetailView(body) {
        return {
            scroll: [body.scrollLeft, body.scrollTop],
            inner: new Map([...body.querySelectorAll('[data-detail-scroll-key]')].map(node =>
                [node.dataset.detailScrollKey, [node.scrollLeft, node.scrollTop]])),
            opened: [...body.querySelectorAll('details')].map(node => node.open),
            focus: body.contains(document.activeElement) ? document.activeElement.dataset.detailScrollKey : null,
        };
    }

    function restoreDetailView(body, view) {
        body.querySelectorAll('[data-detail-scroll-key]').forEach(node => {
            const scroll = view.inner.get(node.dataset.detailScrollKey);
            if (scroll) [node.scrollLeft, node.scrollTop] = scroll;
            if (view.focus && node.dataset.detailScrollKey === view.focus) node.focus({preventScroll: true});
        });
        [body.scrollLeft, body.scrollTop] = view.scroll;
    }

    async function openDetail(nodeId, updating = false) {
        const runId = flow?.id; if (!runId) return;
        // Coalesce background refreshes without invalidating the response the
        // user is waiting for. Explicit node changes still take effect at once.
        if (updating && detailRefreshRequest) {detailRefreshPending = true; return;}
        selected = nodeId; const request = ++detailRequest;
        detailRefreshRequest = request; detailRefreshPending = false;
        hideTooltip();
        if (!detailModal.open) detailModal.show();
        rehomeToast();
        const body = detailModal.querySelector('.if-detail-body');
        if (!updating) {renderedDetail = null; body.replaceChildren(element('p', '', '상세 정보를 불러오는 중…'));}
        try {
            const response = await fetch(`/api/illustration_flow?run=${encodeURIComponent(runId)}&node=${encodeURIComponent(nodeId)}`, {cache: 'no-store'});
            if (!response.ok) throw Error(response.status === 404 ? '새 요청이 접수되어 이전 상세 정보가 종료되었습니다.' : `상세 조회 실패 (${response.status})`);
            const {node: n} = await response.json();
            for (const field of ['label', 'source_call_name']) {
                if (typeof n[field] === 'string') n[field] = n[field].replaceAll('CALL2-SELECT', 'SLOT-SELECTER').replaceAll('CALL2-DETAIL', 'DETAIL').replaceAll('CALL2-KEYVIS', 'KEYVIS').replaceAll('CALL3', 'POST PROCESS');
            }
            if (request !== detailRequest || runId !== flow?.id || !detailModal.open) return;
            const imageUrl = nodeImageUrl(n, flow.status);
            const signature = JSON.stringify([n, activeTab, imageUrl]);
            if (renderedDetail?.runId === runId && renderedDetail.nodeId === nodeId && renderedDetail.signature === signature) {
                const timer = body.querySelector('[data-detail-elapsed]');
                if (timer) timer.textContent = elapsed(n);
                const values = timingDetails(n);
                body.querySelectorAll('[data-detail-timing]').forEach(item => updateText(item, values[item.dataset.detailTiming]));
                return;
            }
            // Capture at response time so scrolling during a slow request wins.
            const view = captureDetailView(body);
            body.replaceChildren(); detailModal.querySelector('h2').textContent = n.label;
            appendTrackedStateTable(n, body);
            const meta = element('dl', 'if-meta');
            const executor = nodeExecutor(n);
            Object.entries({'상태': labels[n.status] || n.status, '처리 주체': executorLabels[executor] || executor,
                ...(n.mask_generation_mode === 'inline' ? {'마스크 생성': '현재 모델에서 자체 생성 · 별도 LLM 호출 없음',
                    '원본 호출': n.source_call_name || '현재 모델', '원본 노드': n.source_node_id || '—'} : {}),
                '모델': (executor === 'llm' ? (window.llmRouteSlotModelName?.(n.llm_slot, true, n.model || '') || n.model) : n.model) || '해당 없음 / 배정 대기', '서비스': n.service || '—', 'LLM 슬롯': n.llm_slot || '—', '라우팅': n.phase || '—', '전체 처리 시간': elapsed(n), ...timingDetails(n), '실행 ID': n.execution_id || n.id, '오류': n.error || '없음'}).forEach(([key,value]) => {
                const item = element('dd', '', value);
                if (key === '전체 처리 시간') item.dataset.detailElapsed = '';
                if (key === 'THINK 시간' || key === '응답 작성 시간') item.dataset.detailTiming = key;
                meta.append(element('dt', '', key), item);
            });
            body.append(meta);
            if (executor === 'llm') body.append(element('p', 'if-subtitle', 'THINK 시간은 첫 추론 수신부터 본문 첫 수신까지, 응답 시간은 본문 첫 수신부터 마지막 수신까지입니다. 진행 중에는 현재 시각까지 갱신합니다. 내부 추론 시간을 추정하지 않으며 추론 수신 기록이 없으면 미수신으로 표시합니다.'));
            const attempts = [];
            for (const event of n.attempts || []) {
                if (!['attempt_start', 'attempt_success', 'attempt_failure'].includes(event.type)) continue;
                const previous = attempts.at(-1);
                if (event.type === 'attempt_start' || !previous || previous.type !== 'attempt_start') attempts.push({...event});
                else Object.assign(previous, event);
            }
            body.append(element('h3', '', `호출·폴백 이력 (${attempts.length}회 시도)`));
            if (!attempts.length) body.append(element('p', 'if-subtitle', '아직 호출 이력이 없거나 LLM을 사용하지 않는 단계입니다.'));
            attempts.forEach((a, index) => {
                const d = element('details', 'if-attempt'); d.open = view.opened[index] || false;
                const state = a.type === 'attempt_failure' ? '실패' : a.type === 'attempt_success' ? '성공' : '진행 중';
                const summary = element('summary'), content = element('div', 'if-attempt-content');
                summary.append(element('span', 'if-attempt-title', `${index + 1}회 · ${a.phase === 'fallback' ? '폴백' : '주 경로'} · ${(window.llmRouteSlotModelName?.(a.slot || a.llm_slot, true, a.model || '') || a.model) || '모델 미확정'} · ${a.slot || a.llm_slot || ''} · ${state}`));
                content.append(element('h3', '', '입력'), detailText(stringify(n.input) || '아직 입력이 없습니다.', `attempt-${index}`));
                if (a.type === 'attempt_failure') {
                    const failure = [...new Set([a.reason, a.error].filter(Boolean))];
                    if (a.raw_response !== undefined && a.raw_response !== null && a.raw_response !== '') failure.push(`실패 응답\n${stringify(a.raw_response)}`);
                    content.append(element('h3', '', '실패 이력'), detailText(failure.join('\n\n') || '실패 사유가 기록되지 않았습니다.', `attempt-${index}-failure`));
                }
                d.append(summary, content);
                body.append(d);
            });
            const output = detailOutput(n);
            body.append(element('h3', '', '입력'), detailText(stringify(n.input) || '아직 입력이 없습니다.', 'input'));
            body.append(element('h3', '', '출력'));
            if (activeTab === 'illustration' && nodePreviewKind(n) === 'image' && imageUrl) {
                const img = element('img', 'if-detail-image'); img.src = imageUrl; img.alt = `${n.label} 생성 결과`; body.append(img);
            }
            if (n.task_key === 'illustration_quality_inspection' && n.inspection_scope === 'overall') appendQualityReviewPanel(n, body, request, runId);
            const outputEmptyText = ['failed', 'cancelled'].includes(n.status)
                ? '실패했지만 기록된 오류나 응답이 없습니다.'
                : '아직 출력이 없습니다.';
            body.append(detailText(stringify(output) || outputEmptyText, 'output'));
            restoreDetailView(body, view);
            renderedDetail = {runId, nodeId, signature};
        } catch (error) {
            console.error('[ILLUST_FLOW] 상세 조회 실패:', error);
            if (request === detailRequest && detailModal.open && (!updating || !renderedDetail)) body.replaceChildren(element('p', '', error.message));
        } finally {
            if (detailRefreshRequest === request) {
                detailRefreshRequest = 0;
                const pending = detailRefreshPending; detailRefreshPending = false;
                if (pending && selected && detailModal.open) void openDetail(selected, true);
            }
        }
    }
    window.receiveIllustrationFlow = receive;
    window.formatLlmStreamTiming = llmTiming;
    window.llmStreamTimingDetails = timingDetails;
    window.receiveVideoFlow = (next, autoOpen) => receive(next, autoOpen, 'video');
    window.receiveVideoInputFlow = (next, autoOpen) => receive(next, autoOpen, 'video_input');
    window.rehomeIllustrationToast = rehomeToast;
    window.refreshIllustrationFlow = refresh;
    window.openIllustrationFlowDeveloperMode = openDeveloperMode;
    window.openWorkflowFlow = async () => {init(); previousFocus = document.activeElement; if (!modal.open) {flowBackdrop.hidden = false; modal.show(); modal.dispatchEvent(new Event('if-open'));} render(); await refresh();};
    window.openIllustrationFlow = async () => {switchTab('illustration'); await window.openWorkflowFlow();};
})();
