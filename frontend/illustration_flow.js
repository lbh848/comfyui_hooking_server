/* Live execution graph; prompt bodies are fetched only when a port is opened. */
(() => {
    let flow = null, modal, detailModal, selected = null, detailRequest = 0;
    let scale = 1, previousFocus = null;
    const labels = {waiting: '대기', processing: '처리 중', completed: '완료', failed: '실패', cancelled: '취소', skipped: '생략'};
    const colors = {waiting: '#94a3b8', processing: '#60a5fa', completed: '#4ade80', failed: '#fb7185', cancelled: '#fbbf24', skipped: '#a78bfa'};
    const element = (tag, className, text) => {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (text !== undefined) el.textContent = text;
        return el;
    };
    const stringify = value => typeof value === 'string' ? value : JSON.stringify(value ?? '', null, 2);
    const elapsed = n => n.started_at ? `${Math.max(0, (n.ended_at || Date.now() / 1000) - n.started_at).toFixed(1)}초` : n.ended_at ? '처리 종료' : '실행 대기';
    function button(text, action) {
        const b = element('button', 'if-button', text); b.type = 'button'; b.onclick = action; return b;
    }
    function init() {
        if (modal) return;
        const style = element('style');
        style.textContent = `
            .if-modal{position:fixed;inset:0;margin:auto;box-sizing:border-box;overflow:hidden;color:var(--text,#e2e8f0);background:var(--bg2,#111827);border:1px solid #64748b66;border-radius:16px;padding:0;width:min(1240px,94vw);max-width:96vw;max-height:calc(100dvh - 32px);box-shadow:0 24px 90px #0009;font:14px/1.5 system-ui,sans-serif}
            .if-modal::backdrop{background:#020617b3;backdrop-filter:blur(3px)}
            .if-header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:18px 22px;border-bottom:1px solid #64748b44}
            .if-header h2{font-size:19px;margin:0}.if-subtitle{font-size:12px;color:var(--text2,#94a3b8);margin-top:4px;overflow-wrap:anywhere}
            .if-actions,.if-legend{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.if-button{border:1px solid #64748b66;border-radius:8px;background:transparent;color:inherit;padding:6px 11px;cursor:pointer}.if-button:hover{background:#64748b33}.if-button:focus-visible,.if-port:focus-visible{outline:3px solid #60a5fa;outline-offset:3px}
            .if-legend{padding:10px 22px;gap:15px;font-size:12px;border-bottom:1px solid #64748b33}.if-legend span::before{content:'●';color:var(--state);margin-right:5px}
            .if-viewport{height:min(65vh,660px);overflow:auto;background-color:var(--bg,#0b1220);background-image:radial-gradient(#94a3b822 1px,transparent 1px);background-size:20px 20px;padding:0;position:relative}
            .if-space{position:relative}.if-canvas{position:relative;transform-origin:0 0}.if-edges{position:absolute;inset:0;overflow:visible;pointer-events:none}
            .if-node{position:absolute;box-sizing:border-box;width:222px;height:98px;border:1px solid #64748b77;border-left:4px solid var(--state);border-radius:11px;background:var(--bg2,#172033);padding:12px 22px 10px 13px;box-shadow:0 4px 15px #0002}
            .if-node[data-status=processing]{box-shadow:0 0 0 2px #60a5fa33,0 0 22px #60a5fa22}.if-node-title{font-weight:650;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.if-node-state{color:var(--state);font-size:12px;margin-top:6px}.if-node-model{font-size:11px;color:var(--text2,#94a3b8);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
            .if-port{position:absolute;right:-9px;top:39px;width:18px;height:18px;border-radius:50%;border:3px solid var(--bg2,#172033);background:var(--state);cursor:pointer;padding:0;box-shadow:0 0 0 1px var(--state)}
            .if-tooltip{position:fixed;z-index:3;max-width:330px;padding:10px 13px;border-radius:9px;background:#0f172a;color:#e2e8f0;border:1px solid #64748b;box-shadow:0 8px 30px #0006;white-space:pre-wrap;pointer-events:none;font-size:12px}
            .if-empty{padding:80px 24px;text-align:center;color:var(--text2,#94a3b8)}.if-empty small{display:block;margin-top:8px}.if-footer{padding:10px 22px;font-size:12px;color:var(--text2,#94a3b8)}
            .if-detail{width:min(900px,94vw)}.if-detail-body{padding:18px 22px;max-height:70vh;overflow:auto}.if-detail-body h3{font-size:14px;margin:18px 0 8px}.if-detail-body pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#64748b14;padding:14px;border-radius:8px;font:12px/1.65 ui-monospace,monospace;margin:0;max-height:340px;overflow:auto}.if-detail-body summary{cursor:pointer;padding:8px 0}.if-meta{display:grid;grid-template-columns:110px 1fr;gap:7px 14px;overflow-wrap:anywhere}.if-meta dt{color:var(--text2,#94a3b8)}.if-meta dd{margin:0}
            @media(max-width:650px){.if-header{align-items:flex-start;padding:14px}.if-actions{justify-content:flex-end}.if-header h2{font-size:16px}.if-meta{grid-template-columns:80px 1fr}}
        `;
        document.head.append(style);
        modal = element('dialog', 'if-modal'); modal.id = 'illustration-flow-modal';
        modal.setAttribute('aria-labelledby', 'if-title');
        const header = element('header', 'if-header'), titleBox = element('div');
        const title = element('h2', '', '삽화 처리 흐름'); title.id = 'if-title';
        titleBox.append(title, element('div', 'if-subtitle', '최신 요청의 실행 상태'));
        const actions = element('div', 'if-actions');
        const resetZoom = button('100%', () => {scale = 1; updateZoomDisplay(); render();});
        resetZoom.id = 'if-zoom-reset';
        resetZoom.title = '현재 확대 비율을 100%로 되돌립니다.';
        actions.append(button('−', () => zoom(-0.15)), button('+', () => zoom(0.15)), resetZoom, button('닫기', () => modal.close()));
        header.append(titleBox, actions);
        const legend = element('div', 'if-legend');
        Object.entries(labels).forEach(([key, label]) => {const s = element('span', '', label); s.style.setProperty('--state', colors[key]); legend.append(s);});
        modal.append(header, legend, element('div', 'if-viewport'), element('footer', 'if-footer', '출력 ●에 마우스를 올리면 요약, 클릭하면 모델·폴백·입력·출력을 확인할 수 있습니다.'));
        modal.addEventListener('close', () => { hideTooltip(); rehomeToast(); previousFocus?.focus(); });
        let ticker;
        modal.addEventListener('close', () => clearInterval(ticker));
        modal.addEventListener('if-open', () => {
            rehomeToast();
            clearInterval(ticker);
            ticker = setInterval(() => {
                const nodes = new Map((flow?.nodes || []).map(n => [n.id, n]));
                modal.querySelectorAll('.if-node').forEach(card => {
                    const n = nodes.get(card.dataset.nodeId);
                    if (n) card.querySelector('.if-node-state').textContent = `${labels[n.status] || n.status} · ${elapsed(n)}`;
                });
            }, 1000);
        });
        detailModal = element('dialog', 'if-modal if-detail'); detailModal.setAttribute('aria-label', '처리 단계 상세');
        const dh = element('header', 'if-header'); dh.append(element('h2', '', '처리 단계 상세'), button('닫기', () => detailModal.close()));
        detailModal.append(dh, element('div', 'if-detail-body'));
        detailModal.addEventListener('close', () => { selected = null; detailRequest++; rehomeToast(); });
        document.body.append(modal, detailModal);
        [modal, detailModal].forEach(dialog => dialog.addEventListener('keydown', event => {
            if (event.key === 'Escape' || event.key === 'Tab') event.stopPropagation();
        }));
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
        const host = detailModal?.open ? detailModal : modal?.open ? modal : toastHome;
        if (toast.parentElement !== host) host.append(toast);
    }
    function hideTooltip() {modal?.querySelector('.if-tooltip')?.remove();}
    function tooltip(n, port) {
        hideTooltip();
        const tip = element('div', 'if-tooltip', `${n.label}\n${labels[n.status] || n.status} · ${elapsed(n)}\n${n.model || (n.kind === 'llm' ? '모델 배정 대기' : '서버 처리')}\n${n.phase === 'fallback' ? '폴백 사용 · ' : ''}${n.error || n.summary || '클릭하여 상세 보기'}`);
        tip.setAttribute('role', 'tooltip'); modal.append(tip);
        const r = port.getBoundingClientRect();
        tip.style.left = `${Math.max(8, Math.min(innerWidth - tip.offsetWidth - 12, r.left - 80))}px`;
        tip.style.top = `${Math.max(8, Math.min(innerHeight - tip.offsetHeight - 12, r.bottom + 10))}px`;
    }
    function render() {
        if (!modal?.open) return;
        hideTooltip();
        const viewport = modal.querySelector('.if-viewport');
        const scroll = [viewport.scrollLeft, viewport.scrollTop];
        const focusedNode = document.activeElement?.dataset?.nodeId;
        viewport.replaceChildren();
        if (!flow) {
            const empty = element('div', 'if-empty', '아직 삽화 요청이 없습니다. 요청이 들어오면 여기에 처리 흐름이 표시됩니다.');
            empty.append(element('small', '', `현재 확대 비율 ${Math.round(scale * 100)}% — 지금 조절한 비율은 다음 그래프에도 적용됩니다.`));
            viewport.append(empty);
            return;
        }
        modal.querySelector('.if-subtitle').textContent = `${flow.label} · ${labels[flow.status] || flow.status} · ${new Date(flow.created_at * 1000).toLocaleString()}`;
        const nodes = flow.nodes || [], positions = new Map(), layers = new Map();
        // Creation order is topological: dependencies exist before their consumer.
        nodes.forEach(n => {
            const depth = Math.max(0, ...(n.dependencies || []).map(id => (positions.get(id)?.depth ?? -1) + 1));
            const lane = layers.get(depth) || 0; layers.set(depth, lane + 1);
            positions.set(n.id, {depth, x: 36 + depth * 290, y: 32 + lane * 136});
        });
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
            const p = positions.get(n.id), card = element('article', 'if-node');
            card.dataset.status = n.status; card.dataset.nodeId = n.id; card.style.setProperty('--state', colors[n.status] || colors.waiting);
            card.style.left = `${p.x}px`; card.style.top = `${p.y}px`;
            const title = element('div', 'if-node-title', n.label); title.title = n.label;
            card.append(title, element('div', 'if-node-state', `${labels[n.status] || n.status} · ${elapsed(n)}`), element('div', 'if-node-model', n.model || (n.kind === 'llm' ? 'LLM' : '서버 처리')));
            const port = button('', () => openDetail(n.id)); port.className = 'if-port'; port.dataset.nodeId = n.id;
            port.setAttribute('aria-label', `${n.label} 출력 상세`);
            port.onmouseenter = port.onfocus = () => tooltip(n, port); port.onmouseleave = port.onblur = hideTooltip;
            card.append(port); canvas.append(card);
        });
        space.append(canvas); viewport.append(space); viewport.scrollLeft = scroll[0]; viewport.scrollTop = scroll[1];
        if (focusedNode) [...canvas.querySelectorAll('.if-port')].find(p => p.dataset.nodeId === focusedNode)?.focus({preventScroll: true});
    }
    async function refresh() {
        const before = flow;
        try {
            const response = await fetch('/api/illustration_flow', {cache: 'no-store'});
            if (!response.ok) throw Error(`흐름 조회 실패 (${response.status})`);
            const next = (await response.json()).flow;
            if (!next && flow === before) {flow = null; if (detailModal?.open) detailModal.close(); render();}
            else receive(next, false);
        } catch (error) {console.error('[ILLUST_FLOW] 최신 상태 조회 실패:', error); if (modal?.open) showToast(error.message, 'error');}
    }
    function receive(next, autoOpen) {
        if (!next) return;
        if (flow && (next.created_at < flow.created_at || (next.id === flow.id && next.revision <= flow.revision))) return;
        const isNew = flow?.id !== next.id;
        flow = next;
        if (isNew && detailModal?.open) detailModal.close();
        if (autoOpen && isNew) {init(); previousFocus = document.activeElement; if (!modal.open) {modal.showModal(); modal.dispatchEvent(new Event('if-open'));}}
        render();
        if (selected && detailModal?.open) openDetail(selected, true);
    }
    async function openDetail(nodeId, updating = false) {
        const runId = flow?.id; if (!runId) return;
        selected = nodeId; const request = ++detailRequest;
        hideTooltip();
        if (!detailModal.open) detailModal.showModal();
        rehomeToast();
        const body = detailModal.querySelector('.if-detail-body');
        if (!updating) body.replaceChildren(element('p', '', '상세 정보를 불러오는 중…'));
        try {
            const response = await fetch(`/api/illustration_flow?run=${encodeURIComponent(runId)}&node=${encodeURIComponent(nodeId)}`, {cache: 'no-store'});
            if (!response.ok) throw Error(response.status === 404 ? '새 삽화 요청이 접수되어 이전 상세 정보가 종료되었습니다.' : `상세 조회 실패 (${response.status})`);
            const {node: n} = await response.json();
            if (request !== detailRequest || runId !== flow?.id || !detailModal.open) return;
            const scroll = body.scrollTop;
            const opened = [...body.querySelectorAll('details')].map(d => d.open);
            body.replaceChildren(); detailModal.querySelector('h2').textContent = n.label;
            const meta = element('dl', 'if-meta');
            Object.entries({'상태': labels[n.status] || n.status, '모델': n.model || '해당 없음 / 배정 대기', '서비스': n.service || '—', 'LLM 슬롯': n.llm_slot || '—', '라우팅': n.phase || '—', '처리 시간': elapsed(n), '실행 ID': n.execution_id || n.id, '오류': n.error || '없음'}).forEach(([key,value]) => meta.append(element('dt', '', key), element('dd', '', value)));
            body.append(meta);
            const attempts = n.attempts || [];
            body.append(element('h3', '', `호출·폴백 이력 (${attempts.filter(a => a.type === 'attempt_start').length}회 시도)`));
            if (!attempts.length) body.append(element('p', 'if-subtitle', '아직 호출 이력이 없거나 LLM을 사용하지 않는 단계입니다.'));
            attempts.forEach((a, index) => {
                const d = element('details'); d.open = opened[index] || false;
                d.append(element('summary', '', `${a.phase === 'fallback' ? '폴백' : '주 경로'} · ${a.model || '모델 미확정'} · ${a.slot || a.llm_slot || ''} · ${a.type}`), element('pre', '', stringify(a))); body.append(d);
            });
            for (const [title, value] of [['입력', n.input], ['출력', n.output]]) body.append(element('h3', '', title), element('pre', '', stringify(value) || '아직 출력이 없습니다.'));
            body.scrollTop = scroll;
        } catch (error) {console.error('[ILLUST_FLOW] 상세 조회 실패:', error); if (request === detailRequest) body.replaceChildren(element('p', '', error.message));}
    }
    window.receiveIllustrationFlow = receive;
    window.rehomeIllustrationToast = rehomeToast;
    window.refreshIllustrationFlow = refresh;
    window.openIllustrationFlow = async () => {init(); previousFocus = document.activeElement; if (!modal.open) {modal.showModal(); modal.dispatchEvent(new Event('if-open'));} render(); await refresh();};
})();
