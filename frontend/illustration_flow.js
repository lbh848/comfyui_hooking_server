/* Live execution graph; prompt bodies are fetched only when a port is opened. */
(() => {
    let flow = null, flowBackdrop, modal, detailModal, developerModal, stopButton, developerButton;
    let selected = null, detailRequest = 0, developerPreviousFocus = null;
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
    const hasDetailValue = value => value !== null && value !== undefined && (typeof value !== 'string' || value.trim() !== '');
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
    function button(text, action) {
        const b = element('button', 'if-button', text); b.type = 'button'; b.onclick = action; return b;
    }
    function init() {
        if (modal) return;
        const style = element('style');
        style.textContent = `
            .if-layer-backdrop{position:fixed;inset:0;z-index:2147483643;background:#020617b3;backdrop-filter:blur(3px)}
            .if-layer-backdrop[hidden]{display:none}
            .if-modal{position:fixed;inset:0;z-index:2147483644;margin:auto;box-sizing:border-box;overflow:hidden;color:var(--text,#e2e8f0);background:var(--bg2,#111827);border:1px solid #64748b66;border-radius:16px;padding:0;width:min(1240px,94vw);max-width:96vw;max-height:calc(100dvh - 32px);box-shadow:0 24px 90px #0009;font:14px/1.5 system-ui,sans-serif}
            .if-modal[open]{display:flex;flex-direction:column}.if-header,.if-tabs,.if-auto-open-settings,.if-legend,.if-footer{flex-shrink:0}
            .if-detail{z-index:2147483645}
            .if-developer{z-index:2147483646;width:min(620px,94vw)}
            .if-header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:18px 22px;border-bottom:1px solid #64748b44}
            .if-header h2{font-size:19px;margin:0}.if-title-row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.if-subtitle{font-size:12px;color:var(--text2,#94a3b8);margin-top:4px;overflow-wrap:anywhere}
            .if-actions,.if-legend,.if-legend-group{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.if-button{border:1px solid #64748b66;border-radius:8px;background:transparent;color:inherit;padding:6px 11px;cursor:pointer}.if-button:hover{background:#64748b33}.if-button:disabled{opacity:.45;cursor:not-allowed}.if-button-danger{border-color:#fb718580;color:#fecdd3}.if-button-danger:not(:disabled):hover{background:#fb71851f}.if-button:focus-visible,.if-port:focus-visible{outline:3px solid #60a5fa;outline-offset:3px}
            .if-developer-button{border-color:#c084fc99;color:#e9d5ff}.if-developer-button:hover{background:#c084fc1f}
            .if-tabs{display:flex;gap:8px;padding:12px 22px 0;border-bottom:1px solid #64748b44}.if-tab{border:0;border-bottom:3px solid transparent;border-radius:8px 8px 0 0;padding:10px 16px;color:var(--text2,#94a3b8)}.if-tab[aria-selected=true]{color:var(--text,#e2e8f0);background:#60a5fa18;border-bottom-color:#60a5fa}.if-modal [hidden]{display:none!important}
            .if-auto-open-settings{padding:10px 22px;border-bottom:1px solid #64748b33}.if-auto-open-toggle{display:flex;align-items:center;gap:9px;width:fit-content;font-size:13px;cursor:pointer}.if-auto-open-toggle input{appearance:none;flex-shrink:0;width:34px;height:20px;margin:0;border:1px solid #64748b;border-radius:12px;background:#475569;cursor:pointer;position:relative}.if-auto-open-toggle input::after{content:'';position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;background:#e2e8f0;transition:transform .15s}.if-auto-open-toggle input:checked{background:#2563eb;border-color:#60a5fa}.if-auto-open-toggle input:checked::after{transform:translateX(14px)}.if-auto-open-toggle input:focus-visible{outline:3px solid #60a5fa;outline-offset:3px}.if-auto-open-state{font-size:12px;color:var(--text2,#94a3b8)}.if-auto-open-help{font-size:12px;color:var(--text2,#94a3b8);margin:5px 0 0;overflow-wrap:anywhere}
            .if-legend{padding:10px 22px;gap:18px;font-size:12px;border-bottom:1px solid #64748b33}.if-legend-group{gap:12px}.if-legend-label{color:var(--text2,#94a3b8);font-weight:650}.if-legend-status .if-legend-item::before{content:'●';color:var(--state);margin-right:5px}.if-legend-executor .if-legend-item::before{content:'';display:inline-block;width:13px;height:13px;border-radius:4px;background:color-mix(in srgb,var(--bg2,#172033) 68%,var(--node-tint) 32%);border:1px solid var(--node-tint);margin-right:6px;vertical-align:-2px}
            .if-viewport{height:min(65vh,660px);min-height:0;flex-shrink:1;overflow:auto;background-color:var(--bg,#0b1220);background-image:radial-gradient(#94a3b822 1px,transparent 1px);background-size:20px 20px;padding:0;position:relative}
            .if-space{position:relative}.if-canvas{position:relative;transform-origin:0 0}.if-edges{position:absolute;inset:0;overflow:visible;pointer-events:none}
            .if-node{position:absolute;box-sizing:border-box;width:222px;height:98px;border:1px solid #64748b77;border-left:4px solid var(--state);border-radius:11px;background:color-mix(in srgb,var(--bg2,#172033) 68%,var(--node-tint,#94a3b8) 32%);padding:12px 22px 10px 13px;box-shadow:0 4px 15px #0002}
            .if-node[data-status=processing]{box-shadow:0 0 0 2px #60a5fa33,0 0 22px #60a5fa22}.if-node-title{font-weight:650;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.if-node-state{color:var(--state);font-size:12px;margin-top:6px}.if-node-model{font-size:11px;color:var(--text2,#94a3b8);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
            .if-port{position:absolute;right:-9px;top:39px;width:18px;height:18px;border-radius:50%;border:3px solid color-mix(in srgb,var(--bg2,#172033) 68%,var(--node-tint,#94a3b8) 32%);background:var(--state);cursor:pointer;padding:0;box-shadow:0 0 0 1px var(--state)}
            .if-node[data-review-node=true] .if-port{width:22px;height:22px;right:-11px;top:37px;border-color:#fef3c7;background:#f59e0b;box-shadow:0 0 0 2px #f59e0b,0 0 16px #f59e0b99}.if-node[data-review-node=true] .if-node-model{color:#fde68a;font-weight:700}
            .if-tooltip{position:fixed;z-index:3;max-width:330px;padding:10px 13px;border-radius:9px;background:#0f172a;color:#e2e8f0;border:1px solid #64748b;box-shadow:0 8px 30px #0006;white-space:pre-wrap;pointer-events:none;font-size:12px}
            .if-empty{padding:80px 24px;text-align:center;color:var(--text2,#94a3b8)}.if-empty small{display:block;margin-top:8px}.if-footer{padding:10px 22px;font-size:12px;color:var(--text2,#94a3b8)}
            .if-detail{width:min(900px,94vw)}.if-detail-body{padding:18px 22px;max-height:70vh;overflow:auto}.if-detail-body h3{font-size:14px;margin:18px 0 8px}.if-detail-body pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#64748b14;padding:14px;border-radius:8px;font:12px/1.65 ui-monospace,monospace;margin:0;max-height:340px;overflow:auto}.if-detail-body summary{cursor:pointer;padding:8px 0}.if-meta{display:grid;grid-template-columns:110px 1fr;gap:7px 14px;overflow-wrap:anywhere}.if-meta dt{color:var(--text2,#94a3b8)}.if-meta dd{margin:0}
            .if-review-panel{border:1px solid #f59e0b88;border-radius:12px;background:#f59e0b0f;padding:16px;margin-bottom:12px}.if-review-title{font-weight:800;font-size:15px;color:#fde68a}.if-review-guide{margin:5px 0 14px;color:var(--text2,#94a3b8);font-size:12px;line-height:1.65}.if-review-images{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:15px}.if-review-figure{margin:0;border:1px solid #64748b55;border-radius:10px;overflow:hidden;background:#02061766}.if-review-image{display:block;width:100%;max-height:520px;object-fit:contain;background:#020617}.if-review-caption{padding:7px 9px;font-size:11px;color:var(--text2,#94a3b8);overflow-wrap:anywhere}.if-review-missing{padding:44px 12px;text-align:center;color:#fecaca;font-size:12px}.if-review-ratings{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px}.if-review-rating{min-width:86px;font-weight:750}.if-review-rating[aria-pressed=true][data-rating=good]{border-color:#4ade80;background:#22c55e33;color:#bbf7d0}.if-review-rating[aria-pressed=true][data-rating=normal]{border-color:#facc15;background:#eab30833;color:#fef08a}.if-review-rating[aria-pressed=true][data-rating=bad]{border-color:#fb7185;background:#e11d4833;color:#fecdd3}.if-review-reason{box-sizing:border-box;width:100%;min-height:78px;resize:vertical;border:1px solid #64748b66;border-radius:8px;background:var(--bg,#0b1220);color:var(--text,#e2e8f0);padding:10px 12px;font:13px/1.5 system-ui,sans-serif}.if-review-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px}.if-review-save{border-color:#f59e0b99;background:#f59e0b22;font-weight:750}.if-review-status{font-size:12px;color:var(--text2,#94a3b8)}
            .if-developer-body{padding:20px 22px;max-height:70vh;overflow:auto}.if-quality-card{border:1px solid #64748b55;border-radius:10px;padding:15px;background:#64748b12}.if-quality-toggle{display:flex;align-items:center;gap:10px;font-weight:700;cursor:pointer}.if-quality-toggle input{width:18px;height:18px;accent-color:#c084fc}.if-quality-copy{margin:9px 0 0;color:var(--text2,#94a3b8);font-size:12px;line-height:1.65}.if-quality-status{margin-top:12px;min-height:1.4em;color:var(--text2,#94a3b8);font-size:12px}.if-developer-footer{display:flex;justify-content:flex-end;padding:12px 22px;border-top:1px solid #64748b44}
            @media(max-width:650px){.if-header{align-items:flex-start;padding:14px}.if-actions{justify-content:flex-end}.if-header h2{font-size:16px}.if-meta{grid-template-columns:80px 1fr}}
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
        modal.append(header, tabs, autoOpenSettings, legend, viewport, element('footer', 'if-footer', '출력 ●을 클릭하면 상세 내용을 볼 수 있습니다. 생성 이미지 자동 검사 노드의 주황색 출력 ●에서는 이미지를 보고 좋음·보통·나쁨을 바로 평가할 수 있습니다.'));
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
                    if (n) card.querySelector('.if-node-state').textContent = nodeState(n);
                });
            }, 1000);
        });
        detailModal = element('dialog', 'if-modal if-detail'); detailModal.setAttribute('aria-label', '처리 단계 상세');
        const dh = element('header', 'if-header'); dh.append(element('h2', '', '처리 단계 상세'), button('닫기', () => detailModal.close()));
        detailModal.append(dh, element('div', 'if-detail-body'));
        detailModal.addEventListener('close', () => { selected = null; detailRequest++; rehomeToast(); });

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
        hideTooltip();
        const executor = nodeExecutor(n);
        const action = n.task_key === 'illustration_quality_inspection'
            ? '클릭하면 생성 이미지가 열리고 여기에서 바로 사람 평가를 할 수 있습니다.'
            : n.error || n.summary || '클릭하여 상세 보기';
        const tip = element('div', 'if-tooltip', `${n.label}\n${executorLabels[executor] || executor} · ${labels[n.status] || n.status} · ${elapsed(n)}\n${n.model || (executor === 'llm' ? '모델 배정 대기' : executor === 'comfy' ? 'ComfyUI' : executor === 'human' ? '사람이 선택한 입력 상태' : '서버 처리')}\n${n.phase === 'fallback' ? '폴백 사용 · ' : ''}${action}`);
        tip.setAttribute('role', 'tooltip'); modal.append(tip);
        const r = port.getBoundingClientRect();
        tip.style.left = `${Math.max(8, Math.min(innerWidth - tip.offsetWidth - 12, r.left - 80))}px`;
        tip.style.top = `${Math.max(8, Math.min(innerHeight - tip.offsetHeight - 12, r.bottom + 10))}px`;
    }
    function render() {
        updateStopButton();
        if (!modal?.open) return;
        hideTooltip();
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
        viewport.replaceChildren();
        if (!flow) {
            modal.querySelector('.if-subtitle').textContent = `${tabLabels[activeTab]} · 최신 요청의 실행 상태`;
            const name = activeTab === 'video_input' ? '영상 입력 개선' : activeTab === 'video' ? '영상' : '삽화';
            const empty = element('div', 'if-empty', `아직 ${name} 요청이 없습니다. 요청이 들어오면 여기에 처리 흐름이 표시됩니다.`);
            if (activeTab === 'video') empty.append(element('small', '', '연출 작성·참조 분석 → 프롬프트 작성·후보 선택 → 영상 생성 → 후처리 → 결과 반환'));
            if (activeTab === 'video_input') empty.append(element('small', '', '입력 상태와 AI 다듬기 결과, 적용·되돌리기 선택을 한 세션 안에 이어서 보존합니다.'));
            empty.append(element('small', '', `현재 확대 비율 ${Math.round(scale * 100)}% — 지금 조절한 비율은 다음 그래프에도 적용됩니다.`));
            viewport.append(empty);
            return;
        }
        modal.querySelector('.if-subtitle').textContent = `${flow.label} · ${labels[flow.status] || flow.status} · ${new Date(flow.created_at * 1000).toLocaleString()}`;
        const nodes = flow.nodes || [], positions = new Map(), layers = new Map();
        const rootColumnX = 36;
        const nodeById = new Map(nodes.map(n => [n.id, n]));
        const isCompactColumnLabel = value => {
            const label = String(value || '');
            return label === 'CHARACTER-RESOLVE' || label.startsWith('CHARACTER-RESOLVE-') ||
                label === 'PROFILE-CONTEXT-CACHE' || label.startsWith('PROFILE-CONTEXT-TRANSLATE') ||
                label === 'PROFILE-RESOLVE' || label.startsWith('PROFILE-RESOLVE-') ||
                label.startsWith('CALL1-BACKTRANSLATE') ||
                label === 'CALL1' || /^CALL1 \d+\/\d+(?:\s|$)/.test(label);
        };
        const isPlanColumnLabel = value => {
            const label = String(value || '');
            return label === 'CALL2-PLAN' || label.startsWith('CALL2-PLAN-');
        };
        const isCompactColumnNode = n => isCompactColumnLabel(n.label) || isCompactColumnLabel(n.call_name);
        const isPlanColumnNode = n => isPlanColumnLabel(n.label) || isPlanColumnLabel(n.call_name);
        const layoutGroup = n => isCompactColumnNode(n)
            ? '__early_compact__'
            : isPlanColumnNode(n)
                ? '__call2_plan__'
                : String(n.layout_group || n.id);
        const membersByGroup = new Map();
        nodes.forEach(n => {
            const group = layoutGroup(n);
            const members = membersByGroup.get(group) || [];
            members.push(n);
            membersByGroup.set(group, members);
        });
        const dependenciesByGroup = new Map();
        nodes.forEach(n => {
            const group = layoutGroup(n);
            const dependencies = dependenciesByGroup.get(group) || new Set();
            (n.dependencies || []).forEach(parentId => {
                const parent = nodeById.get(parentId);
                if (!parent) return;
                const parentGroup = layoutGroup(parent);
                if (parentGroup !== group) dependencies.add(parentGroup);
            });
            dependenciesByGroup.set(group, dependencies);
        });
        const depthByGroup = new Map();
        const resolvingGroups = new Set();
        const groupDepth = group => {
            if (depthByGroup.has(group)) return depthByGroup.get(group);
            if (group === '__early_compact__') {depthByGroup.set(group, 1); return 1;}
            const members = membersByGroup.get(group) || [];
            if (members.some(n => n.kind === 'request')) {depthByGroup.set(group, 0); return 0;}
            if (resolvingGroups.has(group)) {
                console.error('[ILLUST_FLOW] layout_group dependency cycle', group);
                return 2;
            }
            resolvingGroups.add(group);
            const dependencies = [...(dependenciesByGroup.get(group) || [])];
            const minimumDepth = activeTab === 'illustration' ? 2 : dependencies.length ? 1 : 0;
            const depth = Math.max(minimumDepth, ...dependencies.map(parentGroup => groupDepth(parentGroup) + 1));
            resolvingGroups.delete(group);
            depthByGroup.set(group, depth);
            return depth;
        };
        [...membersByGroup.keys()].forEach(groupDepth);

        // A logical stage owns one column. Retries/partial repairs in the same
        // layout_group stack vertically instead of consuming another column.
        // Illustration queue items share one column, while their actual generation
        // stage uses the next logical column. Early images still align with later queue items.
        const insertionOrder = new Map(nodes.map((n, index) => [n.id, index]));
        const orderedNodes = [...nodes].sort((a, b) => {
            const orderA = Number.isFinite(Number(a.layout_order)) ? Number(a.layout_order) : 0;
            const orderB = Number.isFinite(Number(b.layout_order)) ? Number(b.layout_order) : 0;
            return orderA - orderB || insertionOrder.get(a.id) - insertionOrder.get(b.id);
        });
        orderedNodes.forEach(n => {
            const depth = groupDepth(layoutGroup(n));
            const lane = layers.get(depth) || 0;
            layers.set(depth, lane + 1);
            positions.set(n.id, {depth, x: rootColumnX + depth * 290, y: 32 + lane * 136});
        });

        // Keep every logical column top-aligned. Nodes retain their insertion order
        // and stack downward from the same top offset; dependency edges never move
        // a parent or child vertically after the initial column placement.
        const width = Math.max(650, ...[...positions.values()].map(p => p.x + 270));
        const height = Math.max(350, ...[...positions.values()].map(p => p.y + 140));
        const space = element('div', 'if-space'), canvas = element('div', 'if-canvas');
        space.style.width = `${width * scale}px`; space.style.height = `${height * scale}px`;
        canvas.style.width = `${width}px`; canvas.style.height = `${height}px`; canvas.style.transform = `scale(${scale})`;
        const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
        svg.classList.add('if-edges'); svg.setAttribute('width', width); svg.setAttribute('height', height); svg.setAttribute('aria-hidden', 'true');
        nodes.forEach(n => (n.dependencies || []).forEach(id => {
            const a = positions.get(id), b = positions.get(n.id); if (!a || !b) return;
            const path = document.createElementNS(ns, 'path');
            path.setAttribute('d', `M ${a.x + 222} ${a.y + 48} C ${a.x + 257} ${a.y + 48}, ${b.x - 35} ${b.y + 48}, ${b.x} ${b.y + 48}`);
            path.setAttribute('fill', 'none'); path.setAttribute('stroke', colors[n.status] || '#64748b'); path.setAttribute('stroke-width', '1.8'); path.setAttribute('opacity', '.65');
            if (n.status === 'waiting') path.setAttribute('stroke-dasharray', '5 5');
            svg.append(path);
        }));
        canvas.append(svg);
        nodes.forEach(n => {
            const p = positions.get(n.id), card = element('article', 'if-node'), executor = nodeExecutor(n);
            const isQualityReview = n.task_key === 'illustration_quality_inspection';
            card.dataset.status = n.status; card.dataset.nodeId = n.id; card.dataset.executor = executor; card.dataset.reviewNode = String(isQualityReview); card.style.setProperty('--state', colors[n.status] || colors.waiting); card.style.setProperty('--node-tint', executorColors[executor] || executorColors.process);
            card.style.left = `${p.x}px`; card.style.top = `${p.y}px`;
            const title = element('div', 'if-node-title', n.label); title.title = n.label;
            const modelText = isQualityReview
                ? '출력 ● 클릭 → 이미지 평가'
                : n.model || (executor === 'llm' ? 'LLM' : executor === 'comfy' ? 'ComfyUI' : executor === 'human' ? '사람이 선택한 입력 상태' : '서버 처리');
            card.append(title, element('div', 'if-node-state', nodeState(n)), element('div', 'if-node-model', modelText));
            const port = button('', () => openDetail(n.id)); port.className = 'if-port'; port.dataset.nodeId = n.id;
            port.setAttribute('aria-label', isQualityReview ? `${n.label} 이미지 보기 및 사람 평가` : `${n.label} 출력 상세`);
            if (isQualityReview) port.title = '생성 이미지를 보고 좋음·보통·나쁨을 평가합니다.';
            port.onmouseenter = port.onfocus = () => tooltip(n, port); port.onmouseleave = port.onblur = hideTooltip;
            card.append(port); canvas.append(card);
        });
        space.append(canvas); viewport.append(space); viewport.scrollLeft = scroll[0]; viewport.scrollTop = scroll[1];
        if (focusedNode) [...canvas.querySelectorAll('.if-port')].find(p => p.dataset.nodeId === focusedNode)?.focus({preventScroll: true});
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
        panel.append(element('div', 'if-review-title', '생성 이미지 사람 평가'));
        const guide = element('p', 'if-review-guide', '아래 이미지를 직접 확인한 뒤 좋음·보통·나쁨을 선택하고 이유를 간단히 적어 저장하세요. 저장한 이미지와 연결된 전체 LLM 흐름은 자동 정리에서 보호됩니다.');
        panel.append(guide);
        body.append(panel);
        if (!n.history_id) {
            panel.append(element('div', 'if-review-status', n.status === 'processing' || n.status === 'waiting'
                ? '자동 검사가 끝나면 이미지와 평가 입력란이 여기에 표시됩니다.'
                : '이 검사에는 사람 평가를 연결할 기록 ID가 없습니다.'));
            return;
        }
        const reviewUrl = `/api/illustration_quality_inspection/review/${encodeURIComponent(n.history_id)}`;
        const loading = element('div', 'if-review-status', '평가 이미지와 저장된 평가를 불러오는 중…');
        panel.append(loading);
        fetch(reviewUrl, {cache: 'no-store'})
            .then(async response => {
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) throw Error(payload.error || `평가 정보 조회 실패 (${response.status})`);
                return payload;
            })
            .then(payload => {
                if (request !== detailRequest || runId !== flow?.id || !detailModal?.open || !panel.isConnected) return;
                panel.replaceChildren(element('div', 'if-review-title', '생성 이미지 사람 평가'), guide);
                const images = Array.isArray(payload.images) ? payload.images : [];
                const grid = element('div', 'if-review-images');
                if (!images.length) {
                    grid.append(element('div', 'if-review-missing', '이 검사 기록에는 연결된 이미지가 없습니다.'));
                } else {
                    images.forEach((image, index) => {
                        const figure = element('figure', 'if-review-figure');
                        const slot = image.slot ?? index + 1;
                        if (image.image_url) {
                            const img = element('img', 'if-review-image');
                            img.src = image.image_url;
                            img.alt = `생성 이미지 slot ${slot}`;
                            img.loading = 'lazy';
                            img.addEventListener('error', () => {
                                console.error('[ILLUST_QUALITY] 평가 이미지 표시 실패:', image);
                                img.replaceWith(element('div', 'if-review-missing', '이미지 파일을 표시할 수 없습니다.'));
                            }, {once: true});
                            figure.append(img);
                        } else {
                            figure.append(element('div', 'if-review-missing', '이미지 파일이 삭제되었거나 이동되었습니다.'));
                        }
                        figure.append(element('figcaption', 'if-review-caption', `slot ${slot} · ${image.backup_name || '백업명 없음'}`));
                        grid.append(figure);
                    });
                }
                panel.append(grid);
                panel.append(element('div', '', '내 평가'));
                const ratings = element('div', 'if-review-ratings');
                const ratingButtons = [];
                let selectedRating = String(payload.human_evaluation?.rating || '');
                const selectRating = rating => {
                    selectedRating = rating;
                    ratingButtons.forEach(ratingButton => ratingButton.setAttribute('aria-pressed', String(ratingButton.dataset.rating === rating)));
                    saveButton.disabled = !selectedRating;
                };
                for (const [rating, label] of [['good', '좋음'], ['normal', '보통'], ['bad', '나쁨']]) {
                    const ratingButton = button(label, () => selectRating(rating));
                    ratingButton.classList.add('if-review-rating');
                    ratingButton.dataset.rating = rating;
                    ratingButton.setAttribute('aria-pressed', 'false');
                    ratingButtons.push(ratingButton);
                    ratings.append(ratingButton);
                }
                panel.append(ratings);
                const reason = element('textarea', 'if-review-reason');
                reason.value = String(payload.human_evaluation?.reason || '');
                reason.placeholder = '예: 구도와 접촉은 좋지만 복장 일관성이 조금 아쉬움';
                reason.setAttribute('aria-label', '평가 이유');
                panel.append(reason);
                const actions = element('div', 'if-review-actions');
                const saveButton = button('평가 저장', async () => {
                    if (!selectedRating || saveButton.disabled) return;
                    saveButton.disabled = true;
                    status.textContent = '평가와 연결된 LLM 흐름을 저장하는 중…';
                    try {
                        const response = await fetch(reviewUrl, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({rating: selectedRating, reason: reason.value}),
                        });
                        const saved = await response.json().catch(() => ({}));
                        if (!response.ok) throw Error(saved.error || `평가 저장 실패 (${response.status})`);
                        const retained = Number(saved.retained_history_count || 0);
                        const missing = Array.isArray(saved.missing_history_ids) ? saved.missing_history_ids.length : 0;
                        status.textContent = missing
                            ? `저장됨 · LLM 기록 ${retained}건 보존 · 이미 삭제된 ${missing}건은 복구할 수 없습니다.`
                            : `저장됨 · 이미지와 연결된 LLM 기록 ${retained}건을 자동 정리에서 보호합니다.`;
                        status.style.color = missing ? '#fde68a' : '#bbf7d0';
                        qualityInspectionToast('생성 이미지 사람 평가를 저장했습니다.', 'success');
                    } catch (error) {
                        console.error('[ILLUST_QUALITY] 사람 평가 저장 실패:', {historyId: n.history_id, error});
                        status.textContent = `평가 저장 실패: ${error.message || error}`;
                        status.style.color = '#fecdd3';
                        qualityInspectionToast(`생성 이미지 평가 저장 실패: ${error.message || error}`, 'error');
                    } finally {
                        saveButton.disabled = !selectedRating;
                    }
                });
                saveButton.classList.add('if-review-save');
                const status = element('span', 'if-review-status');
                const updatedAt = String(payload.human_evaluation?.updated_at || '');
                status.textContent = updatedAt
                    ? `저장된 평가 · ${updatedAt}`
                    : '평가를 선택하면 저장할 수 있습니다.';
                actions.append(saveButton, status);
                panel.append(actions);
                selectRating(selectedRating);
            })
            .catch(error => {
                console.error('[ILLUST_QUALITY] 사람 평가 정보 조회 실패:', {historyId: n.history_id, error});
                if (request !== detailRequest || runId !== flow?.id || !detailModal?.open || !panel.isConnected) return;
                loading.textContent = n.status === 'processing' || n.status === 'waiting'
                    ? '자동 검사가 끝나면 이미지와 평가 입력란이 여기에 표시됩니다.'
                    : `평가 정보를 불러오지 못했습니다: ${error.message || error}`;
                loading.style.color = n.status === 'processing' || n.status === 'waiting' ? '' : '#fecdd3';
            });
    }
    async function openDetail(nodeId, updating = false) {
        const runId = flow?.id; if (!runId) return;
        selected = nodeId; const request = ++detailRequest;
        hideTooltip();
        if (!detailModal.open) detailModal.show();
        rehomeToast();
        const body = detailModal.querySelector('.if-detail-body');
        if (!updating) body.replaceChildren(element('p', '', '상세 정보를 불러오는 중…'));
        try {
            const response = await fetch(`/api/illustration_flow?run=${encodeURIComponent(runId)}&node=${encodeURIComponent(nodeId)}`, {cache: 'no-store'});
            if (!response.ok) throw Error(response.status === 404 ? '새 요청이 접수되어 이전 상세 정보가 종료되었습니다.' : `상세 조회 실패 (${response.status})`);
            const {node: n} = await response.json();
            if (request !== detailRequest || runId !== flow?.id || !detailModal.open) return;
            const scroll = body.scrollTop;
            const opened = [...body.querySelectorAll('details')].map(d => d.open);
            body.replaceChildren(); detailModal.querySelector('h2').textContent = n.label;
            const meta = element('dl', 'if-meta');
            const executor = nodeExecutor(n);
            Object.entries({'상태': labels[n.status] || n.status, '처리 주체': executorLabels[executor] || executor, '모델': n.model || '해당 없음 / 배정 대기', '서비스': n.service || '—', 'LLM 슬롯': n.llm_slot || '—', '라우팅': n.phase || '—', '처리 시간': elapsed(n), '실행 ID': n.execution_id || n.id, '오류': n.error || '없음'}).forEach(([key,value]) => meta.append(element('dt', '', key), element('dd', '', value)));
            body.append(meta);
            const attempts = n.attempts || [];
            body.append(element('h3', '', `호출·폴백 이력 (${attempts.filter(a => a.type === 'attempt_start').length}회 시도)`));
            if (!attempts.length) body.append(element('p', 'if-subtitle', '아직 호출 이력이 없거나 LLM을 사용하지 않는 단계입니다.'));
            attempts.forEach((a, index) => {
                const d = element('details'); d.open = opened[index] || false;
                d.append(element('summary', '', `${a.phase === 'fallback' ? '폴백' : '주 경로'} · ${a.model || '모델 미확정'} · ${a.slot || a.llm_slot || ''} · ${a.type}`), element('pre', '', stringify(a))); body.append(d);
            });
            const output = detailOutput(n);
            body.append(element('h3', '', '입력'), element('pre', '', stringify(n.input) || '아직 입력이 없습니다.'));
            body.append(element('h3', '', '출력'));
            if (n.task_key === 'illustration_quality_inspection') appendQualityReviewPanel(n, body, request, runId);
            const outputEmptyText = ['failed', 'cancelled'].includes(n.status)
                ? '실패했지만 기록된 오류나 응답이 없습니다.'
                : '아직 출력이 없습니다.';
            body.append(element('pre', '', stringify(output) || outputEmptyText));
            body.scrollTop = scroll;
        } catch (error) {console.error('[ILLUST_FLOW] 상세 조회 실패:', error); if (request === detailRequest) body.replaceChildren(element('p', '', error.message));}
    }
    window.receiveIllustrationFlow = receive;
    window.receiveVideoFlow = (next, autoOpen) => receive(next, autoOpen, 'video');
    window.receiveVideoInputFlow = (next, autoOpen) => receive(next, autoOpen, 'video_input');
    window.rehomeIllustrationToast = rehomeToast;
    window.refreshIllustrationFlow = refresh;
    window.openIllustrationFlowDeveloperMode = openDeveloperMode;
    window.openWorkflowFlow = async () => {init(); previousFocus = document.activeElement; if (!modal.open) {flowBackdrop.hidden = false; modal.show(); modal.dispatchEvent(new Event('if-open'));} render(); await refresh();};
    window.openIllustrationFlow = async () => {switchTab('illustration'); await window.openWorkflowFlow();};
})();
