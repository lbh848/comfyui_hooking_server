/* User-triggered, read-only reasoning diagnostics for the saved illustration LLMs. */
(() => {
    'use strict';
    const results = new Map();
    const pending = new Set();
    let panel = null, previewRoot = null, savedConfig = {}, slots = [], checkingAll = false, allError = '';
    let tooltip = null, tooltipOwner = null;
    const validSlot = slot => /^llm(?:[1-9]|10)$/.test(slot);
    const labels = {observed: 'Think 사용 확인', zero: '이번 요청 0토큰', unknown: 'Think 미사용'};

    function illustrationThinkingSlots(names, config) {
        const assigned = new Map();
        const add = (slot, name, role) => {
            if (!validSlot(slot)) return;
            if (!assigned.has(slot)) assigned.set(slot, new Set());
            assigned.get(slot).add(`${name} (${role})`);
        };
        for (const name of new Set(names)) {
            const [task] = getIllustrationPreviewTasks([name]);
            if (!task) continue;
            const route = config.llm_routing?.[task] || {};
            const primary = route.primary;
            if (!validSlot(primary)) {
                console.error('[THINK_CHECK] 메인 슬롯 미설정:', {task, route});
                continue;
            }
            add(primary, name, '메인');
            add(llmRouteResolvedFallback(route, primary), name, '폴백');
        }
        return [...assigned].sort(([a], [b]) => Number(a.slice(3)) - Number(b.slice(3)))
            .map(([slot, tasks]) => ({slot, tasks: [...tasks]}));
    }

    function signature(slot, config) {
        const suffix = slot === 'llm1' ? '' : slot.slice(3);
        const fields = ['service', 'model', 'api_key', 'url', 'reasoning_preset',
            'reasoning_effort', 'reasoning_budget_tokens', 'temperature', 'custom_body',
            'custom_headers', 'stream', 'max_tokens', 'max_concurrency',
            'stream_idle_timeout_seconds', 'vision_compress', 'gemini_base64', 'pdf_prompt'];
        return JSON.stringify(fields.map(field => [field, config[`llm_${field}${suffix}`]]));
    }

    function element(tag, text, parent) {
        const node = document.createElement(tag);
        if (text !== undefined) node.textContent = text;
        if (parent) parent.append(node);
        return node;
    }

    function currentResult(slot) {
        const result = results.get(slot);
        if (pending.has(slot)) return {status: 'checking', label: '대기·검사 중…'};
        return result?.signature === signature(slot, savedConfig) ? result : null;
    }

    function nodeHealth(name, config, resultForSlot) {
        const [task] = getIllustrationPreviewTasks([name]);
        if (!task) return null;
        const route = config.llm_routing?.[task] || {};
        const primary = route.primary;
        if (!validSlot(primary)) {
            console.error('[THINK_CHECK] 메인 슬롯 미설정:', {task, route});
            return null;
        }
        const fallback = llmRouteResolvedFallback(route, primary);
        const main = resultForSlot(primary), backup = validSlot(fallback) ? resultForSlot(fallback) : null;
        const connection = (slot, role, result) => {
            const suffix = slot === 'llm1' ? '' : slot.slice(3);
            const model = llmRouteSlotModelName(slot, true, config[`llm_model${suffix}`] || '', config);
            return `${role}: ${slot.toUpperCase()} · ${model || '(모델 미설정)'}\n${result?.label || '미검사'}`;
        };
        const details = [connection(primary, '메인', main), validSlot(fallback)
            ? connection(fallback, '폴백', backup) : '폴백: 설정 없음'];
        const problems = [];
        if (main?.status === 'observed') problems.push('메인 LLM에서 Think 사용이 확인되었습니다.');
        if (main?.status === 'error') problems.push('메인 LLM 검사 요청이 실패했습니다.');
        if (!validSlot(fallback)) problems.push('폴백이 설정되지 않았습니다.');
        else if (fallback === primary) problems.push('폴백이 메인과 같은 연결입니다.');
        else {
            const suffix = fallback === 'llm1' ? '' : fallback.slice(3);
            if (!String(config[`llm_model${suffix}`] || '').trim()) problems.push('폴백 모델이 설정되지 않았습니다.');
            if (backup?.status === 'error') problems.push('폴백 LLM 검사 요청이 실패했습니다.');
        }
        // Badges summarize completed checks; missing reasoning evidence counts as unused.
        const checked = result => result && result.status !== 'checking';
        let color = 'gray', summary = '검사 대기';
        if (checked(main) || checked(backup)) {
            if (problems.length) { color = 'red'; summary = '메인 Think 또는 연결 설정 확인 필요'; }
            else if (backup?.status === 'observed') { color = 'yellow'; summary = '폴백에서 Think 사용 확인'; }
            else if (['zero', 'unknown'].includes(main?.status) && ['zero', 'unknown'].includes(backup?.status)) {
                color = 'green'; summary = 'Think 미사용 · 폴백 응답 확인';
            }
        }
        return {color, summary, details: [name, summary, ...problems, ...details,
            '추론 정보가 없는 검사 응답은 Think 미사용으로 취급합니다.'].join('\n\n')};
    }

    function hideTooltip() {
        if (tooltip) tooltip.hidden = true;
        tooltipOwner?.removeAttribute('aria-describedby');
        tooltipOwner = null;
    }

    function showTooltip(badge) {
        if (!tooltip?.isConnected) {
            tooltip = element('div', undefined, document.body);
            tooltip.id = 'illust-thinking-node-tooltip';
            tooltip.className = 'illust-thinking-node-tooltip';
            tooltip.setAttribute('role', 'tooltip');
        }
        tooltipOwner?.removeAttribute('aria-describedby');
        tooltipOwner = badge;
        tooltip.textContent = badge.dataset.thinkingTooltip;
        tooltip.hidden = false;
        badge.setAttribute('aria-describedby', tooltip.id);
        const bounds = badge.getBoundingClientRect();
        tooltip.style.left = `${Math.max(8, Math.min(bounds.right + 8, innerWidth - tooltip.offsetWidth - 8))}px`;
        tooltip.style.top = `${Math.max(8, Math.min(bounds.top, innerHeight - tooltip.offsetHeight - 8))}px`;
    }

    function updateBadges() {
        if (!previewRoot) return;
        if (tooltipOwner && !tooltipOwner.isConnected) hideTooltip();
        for (const node of previewRoot.querySelectorAll('[data-illust-preview-node]')) {
            let badge = node.querySelector('.illust-thinking-node-badge');
            const health = node.dataset.enabled === 'true'
                ? nodeHealth(node.dataset.illustPreviewNode, savedConfig, currentResult) : null;
            if (!health || !results.size) {
                if (badge === tooltipOwner) hideTooltip();
                badge?.remove();
                continue;
            }
            if (!badge) {
                badge = element('button', undefined, node);
                badge.type = 'button'; badge.title = '';
                badge.className = 'illust-thinking-node-badge';
                badge.onmouseenter = badge.onfocus = () => showTooltip(badge);
                badge.onmouseleave = badge.onblur = hideTooltip;
                badge.onclick = () => tooltipOwner === badge ? hideTooltip() : showTooltip(badge);
            }
            badge.dataset.thinkingHealth = health.color;
            badge.dataset.thinkingTooltip = health.details;
            badge.textContent = {red: '!', yellow: '!', green: '✓', gray: '?'}[health.color];
            badge.setAttribute('aria-label', `${node.dataset.illustPreviewNode} · ${health.summary}`);
            if (tooltipOwner === badge) showTooltip(badge);
        }
    }

    function render() {
        updateBadges();
        if (!panel) return;
        panel.replaceChildren();
        const toolbar = element('div', undefined, panel);
        toolbar.style.cssText = 'display:flex;align-items:center;gap:12px;flex-wrap:wrap';
        element('h4', '삽화 LLM Think 확인', toolbar).style.margin = '0';
        const all = element('button', checkingAll ? '전체 확인 중…' : '모두 확인', toolbar);
        all.type = 'button'; all.disabled = checkingAll || pending.size > 0 || !slots.length;
        all.onclick = async () => {
            checkingAll = true; allError = ''; render();
            try {
                await loadCurrentConfig(true);
                refresh(document.getElementById('illust-node-preview'), currentConfig);
                const targets = slots.map(item => item.slot);
                for (const slot of targets) await check(slot, false);
            } catch (error) {
                console.error('[THINK_CHECK] 전체 확인 실패:', error);
                allError = `요청 실패: ${error.message}`;
            } finally { checkingAll = false; render(); }
        };
        element('p', '현재 저장된 연결 설정으로 짧은 계산 요청을 보냅니다. 설정은 변경하지 않습니다. API 사용량이 발생합니다.', panel);
        element('p', '추론 블록이나 추론 토큰이 오지 않은 검사 응답은 Think 미사용으로 취급합니다. 노드 왼쪽 뱃지에 마우스를 올리면 메인·폴백 결과를 볼 수 있습니다.', panel);
        if (allError) element('p', allError, panel).setAttribute('role', 'alert');
        for (const {slot, tasks} of slots) {
            const row = element('div', undefined, panel);
            row.dataset.thinkingSlot = slot;
            row.style.cssText = 'display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 0;border-top:1px solid var(--border)';
            const info = element('div', undefined, row);
            info.style.cssText = 'flex:1;min-width:200px';
            const suffix = slot === 'llm1' ? '' : slot.slice(3);
            const model = llmRouteSlotModelName(slot, true, savedConfig[`llm_model${suffix}`] || '', savedConfig) || '(모델 미설정)';
            element('strong', `${slot.toUpperCase()} · ${model}`, info);
            const roles = element('div', tasks.join(', '), info);
            roles.style.cssText = 'color:var(--text2);font-size:.78rem';
            const current = currentResult(slot);
            const status = element('div', pending.has(slot) ? '대기·검사 중…' : (current?.label || '미검사'), row);
            status.dataset.thinkingStatus = pending.has(slot) ? 'checking' : (current?.status || 'idle');
            status.setAttribute('role', 'status');
            const button = element('button', 'Think 확인', row);
            button.type = 'button'; button.disabled = checkingAll || pending.has(slot);
            button.onclick = () => check(slot);
        }
    }

    async function readEvents(response, onEvent) {
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
        if (!response.body) throw new Error('검사 응답 본문이 없습니다.');
        const reader = response.body.getReader(), decoder = new TextDecoder();
        let buffer = '';
        const dispatch = block => {
            const lines = block.split('\n');
            const type = lines.find(line => line.startsWith('event:'))?.slice(6).trim();
            const data = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
            if (type && data) onEvent(type, JSON.parse(data));
        };
        try {
            while (true) {
                const {value, done} = await reader.read();
                buffer += done ? decoder.decode() : decoder.decode(value, {stream: true});
                buffer = buffer.replaceAll('\r\n', '\n');
                let end;
                while ((end = buffer.indexOf('\n\n')) >= 0) {
                    dispatch(buffer.slice(0, end)); buffer = buffer.slice(end + 2);
                }
                if (done) { if (buffer.trim()) dispatch(buffer); break; }
            }
        } finally { reader.releaseLock(); }
    }

    async function check(slot, reload = true) {
        if (pending.has(slot)) return;
        pending.add(slot); render();
        let requestSignature = signature(slot, savedConfig);
        try {
            if (reload) {
                await loadCurrentConfig(true);
                refresh(document.getElementById('illust-node-preview'), currentConfig);
            }
            requestSignature = signature(slot, savedConfig);
            let done = null;
            const response = await fetch('/api/llm/test_stream', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({target: slot, thinking_check: true}),
            });
            await readEvents(response, (type, data) => {
                if (type === 'error' || type === 'cancelled') throw new Error(data.error || data.reason || '검사 실패');
                if (type === 'done') done = data;
            });
            if (!done) throw new Error('검사 결과 없이 연결이 종료되었습니다.');
            const status = done.thinking_status || 'unknown';
            const evidence = done.reasoning_tokens != null ? ` · 추론 ${done.reasoning_tokens}토큰`
                : done.reasoning_chars > 0 ? ' · 추론 데이터 수신' : '';
            const elapsed = Number(done.elapsed);
            const duration = Number.isFinite(elapsed) ? ` · ${elapsed.toFixed(2)}초` : '';
            results.set(slot, {signature: requestSignature, status,
                label: `${labels[status] || labels.unknown}${evidence}${duration}`});
        } catch (error) {
            console.error('[THINK_CHECK] 검사 실패:', {slot, error});
            results.set(slot, {signature: requestSignature, status: 'error', label: `요청 실패 · ${error.message}`});
        } finally { pending.delete(slot); render(); }
    }

    function refresh(preview, config) {
        if (!preview) return;
        previewRoot = preview;
        savedConfig = config || {};
        const names = [...preview.querySelectorAll('[data-illust-preview-node][data-enabled="true"]')]
            .map(node => node.dataset.illustPreviewNode);
        slots = illustrationThinkingSlots(names, savedConfig);
        const anchor = preview.closest('.illust-flow-preview');
        if (!anchor) return;
        if (!panel?.isConnected) {
            panel = element('section'); panel.dataset.illustThinkingChecks = '';
            panel.className = 'illust-thinking-checks';
            panel.style.cssText = 'margin-top:16px;padding:14px;border:1px solid var(--border);border-radius:8px;font-size:.85rem';
            anchor.after(panel);
        }
        render();
    }

    window.getIllustrationThinkingSlots = illustrationThinkingSlots;
    window.getIllustrationThinkingNodeHealth = nodeHealth;
    window.refreshIllustrationThinkingChecks = refresh;
    window.runIllustrationThinkingCheck = check;
    document.addEventListener('scroll', hideTooltip, true);
    document.addEventListener('keydown', event => {if (event.key === 'Escape') hideTooltip();});
})();
