/* Settings-only explanation of the illustration pipeline. No generation requests. */
(() => {
    'use strict';
    const transferTime = 0.3;
    const packetSpeed = 400;
    const nodeTime = 0.2;

    const taskKeys = {
        'MANGA-CHARACTER-RESOLVE': 'illustration_character_resolve',
        'PLANNER-MANGA': 'illustration_call2_plan',
        'DETAIL-MANGA': 'illustration_call2',
        'CHARACTER-RESOLVE': 'illustration_character_resolve',
        'PROFILE-CONTEXT-TRANSLATE': 'illustration_profile_context_translate',
        'PROFILE-RESOLVE': 'illustration_profile_resolve',
        'OUTFIT': 'illustration_call1',
        'CALL1': 'illustration_call1',
        'TIME & PLACE': 'illustration_time_place',
        'CALL2-PLAN': 'illustration_call2_plan',
        'CALL2-DETAIL': 'illustration_call2',
        'CALL2-KEYVIS': 'illustration_call2_keyvis',
        'CALL3': 'illustration_call3',
        'MULTI-CHAR-MASK': 'illustration_multi_char_mask',
        'ORIGINAL-ASSET': 'illustration_original_asset',
        'ORIGINAL-ASSET-RECOVERY': 'illustration_original_asset_recovery',
        '생성 이미지 자동 검사': 'illustration_quality_inspection',
        '생성 이미지 전체 품질 검사': 'illustration_quality_inspection',
    };

    function getIllustrationPreviewLLM(name, config = {}) {
        const task = taskKeys[name] || taskKeys[Object.keys(taskKeys).find(key => name.startsWith(key + ' '))];
        if (!task) return null;
        const slot = config.llm_routing?.[task]?.primary;
        if (!/^llm(?:[1-9]|10)$/.test(slot)) {
            console.error('[ILLUST_PREVIEW] 작업의 LLM 슬롯이 설정되지 않았습니다:', {task, slot});
            return null;
        }
        const suffix = slot === 'llm1' ? '' : slot.slice(3);
        const key = `llm_max_concurrency${suffix}`;
        const raw = config[key] ?? 1;
        let limit = Number(raw);
        if (typeof raw === 'boolean' || !Number.isInteger(limit) || limit < 1 || limit > 20) {
            console.error('[ILLUST_PREVIEW] LLM 동시 요청 상한이 올바르지 않아 1을 사용합니다:', {slot, key, value: raw});
            limit = 1;
        }
        return {slot, limit};
    }

    function buildIllustrationPreviewSimulation(plan, transferTimes = new Map()) {
        // Brief node flashes keep the preview focused on data flow.
        // Dependencies and resource limits still come from the execution plan.
        const jobs = plan.jobs.map(job => ({...job, parents: [...job.parents], start: Infinity, end: Infinity}));
        const byId = new Map(jobs.map(job => [job.id, job]));
        const pending = new Set(jobs);
        let active = [], time = 0;
        const readyAt = job => {
            const required = [...new Set([...job.parents, ...(job.data_parents || [])])].map(id => {
                const parent = byId.get(id);
                return parent.end + (transferTimes.get(JSON.stringify([parent.card, job.card])) ?? transferTime);
            });
            if (job.any_parents?.length) required.push(Math.min(...job.any_parents.map(id => {
                const parent = byId.get(id);
                return parent.end + (transferTimes.get(JSON.stringify([parent.card, job.card])) ?? transferTime);
            })));
            return required.length ? Math.max(...required) : 0;
        };
        while (pending.size) {
            active = active.filter(job => job.end > time);
            const ready = [...pending].filter(job => readyAt(job) <= time)
                .sort((a, b) => a.priority - b.priority);
            ready.forEach(job => {
                const resources = job.resources.find(option => option.every(resource =>
                    active.filter(other => other.usedResources.includes(resource)).length < plan.capacities[resource]));
                if (!resources) return;
                job.usedResources = resources;
                job.start = time;
                job.end = time + nodeTime;
                if (job.any_parents?.length) {
                    const first = job.any_parents.reduce((a, b) => {
                        const left = byId.get(a), right = byId.get(b);
                        return left.end + (transferTimes.get(JSON.stringify([left.card, job.card])) ?? transferTime)
                            <= right.end + (transferTimes.get(JSON.stringify([right.card, job.card])) ?? transferTime) ? a : b;
                    });
                    job.parents.push(first);
                }
                active.push(job);
                pending.delete(job);
            });
            if (pending.size) {
                const next = [...active.map(job => job.end), ...[...pending].map(readyAt)]
                    .filter(nextTime => Number.isFinite(nextTime) && nextTime > time);
                if (!next.length) {
                    console.error('[ILLUST_PREVIEW] 실행 계획을 진행할 수 없음', {pending: [...pending], capacities: plan.capacities});
                    return {jobs, blocked: [...pending].map(job => job.id),
                        duration: Math.max(1, ...jobs.filter(job => Number.isFinite(job.end)).map(job => job.end)) + transferTime};
                }
                time = Math.min(...next);
            }
        }
        return {jobs, blocked: [], duration: Math.max(1, ...jobs.map(job => job.end)) + transferTime};
    }

    function buildIllustrationPreviewConnections(model, plan, transferTimes = new Map()) {
        const byId = new Map(model.jobs.map(job => [job.id, job]));
        const transfers = model.jobs.flatMap(job => (job.data_parents ?? job.parents).map(parentId =>
            ({from: byId.get(parentId).card, to: job.card,
                start: byId.get(parentId).end, end: byId.get(parentId).end
                    + (transferTimes.get(JSON.stringify([byId.get(parentId).card, job.card])) ?? transferTime)})));
        const nodes = new Map(plan.nodes.map(node => [node.id, node]));
        const links = plan.connections ?? [...new Map(transfers.map(edge =>
            [JSON.stringify([edge.from, edge.to]), {from: edge.from, to: edge.to}])).values()];
        return links.filter(edge => edge.from !== edge.to).map(edge => ({
            from: nodes.get(edge.from), to: nodes.get(edge.to),
            transfers: transfers.filter(transfer => transfer.from === edge.from && transfer.to === edge.to
                && Number.isFinite(transfer.start)),
        }));
    }

    function renderIllustrationPreviewPlan(root, plan, activeGroup) {
        const escape = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
        const labels = {call1_select: 'OUTFIT + TIME & PLACE',
            call2_plan: 'PLAN 배치', call2_detail: 'DETAIL 배치', keyvis: 'KEYVIS',
            original_asset: 'Original Asset', profile_resolve: 'PROFILE-RESOLVE',
            illustration_images: '삽화', illustration_postprocess: '이미지 후처리',
            illustration_inspection: '생성 이미지 자동 검사'};
        const nodeHTML = node => `<article class="illust-flow-preview-node${node.enabled ? '' : ' is-disabled'}${node.group === activeGroup || (activeGroup === 'call2_plan_detail' && ['call2_plan', 'call2_detail'].includes(node.group)) ? ' is-current' : ''}"
            data-illust-preview-node="${escape(node.name)}" data-preview-id="${escape(node.id)}"
            data-enabled="${node.enabled}" data-condition="${escape(node.condition)}"
            data-batch-index="${node.index}" data-batch-count="${node.count}">
            <strong>${escape(node.name.replaceAll('CALL2-SELECT', 'SLOT-SELECTER').replaceAll('CALL2-PLAN', 'PLAN').replaceAll('CALL2-DETAIL', 'DETAIL').replaceAll('CALL2-KEYVIS', 'KEYVIS').replaceAll('CALL3', 'POST PROCESS'))}${node.count > 1 ? `<small class="illust-flow-preview-batch-index">${node.index}/${node.count}</small>` : ''}<small class="illust-preview-activation" data-activation="${node.enabled ? 'on' : 'off'}">사용 ${node.enabled ? 'ON' : 'OFF'}</small></strong>
            <span>${node.enabled ? '' : '생략 · '}${escape(node.description)}</span>
            ${node.condition && node.enabled ? `<span class="illust-preview-condition">조건: ${escape(node.condition)}</span>` : ''}
            ${node.llm ? `<span class="illust-preview-llm">${node.llm.slot.toUpperCase()} · 슬롯 공유 상한 ${node.llm.limit}개</span>` : ''}
            </article>`;
        root.innerHTML = plan.steps.map((title, index) => {
            const groups = new Map();
            plan.nodes.filter(node => node.step === index + 1).forEach(node => {
                const key = labels[node.group] ? node.group : node.id;
                if (!groups.has(key)) groups.set(key, []);
                groups.get(key).push(node);
            });
            const content = [...groups].map(([key, nodes]) => {
                const cardIds = new Set(nodes.map(node => node.id));
                const count = plan.jobs.filter(job => cardIds.has(job.card)).length;
                return labels[key] ? `<section class="illust-flow-preview-parallel" data-illust-preview-group="${escape(key)}" role="group" aria-label="${escape(labels[key])}">
                 <h5 class="illust-flow-preview-parallel-title">${escape(labels[key])} · 큐 작업 ${count}개</h5>
                 ${nodes.map(nodeHTML).join('')}</section>` : nodes.map(nodeHTML).join('');
            }).join('');
            const stepLabel = index < 6 ? `STEP ${index + 1}` : title;
            return `<section class="illust-flow-preview-step" aria-label="${escape(stepLabel)}"><h4 class="illust-flow-preview-step-title">${escape(stepLabel)}${index < 6 ? ` · ${escape(title)}` : ''}</h4>${content}</section>`;
        }).join('');
        const container = root.closest('.illust-flow-preview');
        container.querySelector('[data-illust-preview-summary]').textContent = plan.summary;
        container.querySelector('.illust-flow-preview-note').textContent = plan.notes.join(' ');
        refresh(root, plan);
    }

    let current = null;
    const frameInterval = 1000 / 60;
    const ns = 'http://www.w3.org/2000/svg';
    const stateLabels = {waiting: '대기', processing: '처리 중', completed: '완료', skipped: '조건 생략', disabled: 'OFF'};
    const element = (tag, className, text) => {
        const node = document.createElement(tag);
        node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    };
    const svgElement = (tag, attributes) => {
        const node = document.createElementNS(ns, tag);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        return node;
    };
    const isVisible = () => current?.root.isConnected && !document.hidden
        && current.root.getClientRects().length > 0;

    function drawConnections() {
        if (!current) return;
        if (!isVisible()) {pause(); return;}
        const {root, cards, plan, svg} = current;
        const bounds = root.getBoundingClientRect();
        // Read geometry before changing the SVG. Ancestor style notifications
        // often leave the graph unchanged, so do not rebuild it in that case.
        const width = root.clientWidth, height = root.clientHeight;
        const cardBounds = new Map([...cards].map(([id, card]) => [id, card.node.getBoundingClientRect()]));
        const layout = JSON.stringify([width, height, [...cardBounds].map(([id, rect]) =>
            [id, rect.left - bounds.left, rect.top - bounds.top, rect.width, rect.height])]);
        if (current.layout === layout) {play(); return;}
        current.layout = layout;
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.replaceChildren();
        const marker = svgElement('marker', {id: 'illust-preview-arrow', viewBox: '0 0 10 10',
            refX: '8', refY: '5', markerWidth: '5', markerHeight: '5', orient: 'auto-start-reverse'});
        marker.append(svgElement('path', {d: 'M 0 0 L 10 5 L 0 10 z', fill: 'currentColor'}));
        const defs = svgElement('defs', {}); defs.append(marker); svg.append(defs);
        current.edges = [];
        current.connections.forEach(({from: parent, to: job, transfers}) => {
            const a = cardBounds.get(parent.id);
            const b = cardBounds.get(job.id);
            const ax = a.right - bounds.left, ay = a.top + a.height / 2 - bounds.top;
            const bx = b.left - bounds.left, by = b.top + b.height / 2 - bounds.top;
            const sameColumn = Math.abs(a.left - b.left) < 40;
            let d;
            if (sameColumn) {
                const end = b.right - bounds.left, lane = Math.max(ax, end) + 6;
                d = `M ${ax} ${ay} C ${lane} ${ay}, ${lane} ${by}, ${end} ${by}`;
            } else {
                // Match the FLOW graph's direct cubic connections, keeping
                // the handles inside the preview's narrower column gaps.
                const bend = Math.min(35, Math.abs(bx - ax) / 2);
                d = `M ${ax} ${ay} C ${ax + bend} ${ay}, ${bx - bend} ${by}, ${bx} ${by}`;
            }
            const path = svgElement('path', {d, class: 'illust-preview-edge', 'marker-end': 'url(#illust-preview-arrow)',
                'data-from-node': parent.id, 'data-to-node': job.id, 'data-simulated': String(transfers.length > 0)});
            const packet = svgElement('circle', {r: '3', class: 'illust-preview-packet'});
            svg.append(path, packet);
            current.edges.push({path, packet, transfers});
        });
        // Measure the rendered curve so every dot travels
        // at the same pixel speed and the destination waits for its arrival.
        const transferTimes = new Map(current.edges.map(edge =>
            [JSON.stringify([edge.path.dataset.fromNode, edge.path.dataset.toNode]), edge.path.getTotalLength() / packetSpeed]));
        current.model = buildIllustrationPreviewSimulation(plan, transferTimes);
        cards.forEach(card => {card.jobs = [];});
        current.model.jobs.forEach(job => {cards.get(job.card).jobs.push(job);});
        current.groupStatuses.forEach(group => {
            const slots = new Set(group.jobs.map(job => job.llm.slot));
            group.slotJobs = current.model.jobs.filter(job => job.llm && slots.has(job.llm.slot));
        });
        current.connections = buildIllustrationPreviewConnections(current.model, plan, transferTimes);
        current.connections.forEach((connection, index) => {current.edges[index].transfers = connection.transfers;});
        current.time %= current.model.duration;
        if (current.playing) paint();
        else play();
    }

    function paint() {
        if (!current) return;
        const {time, model, cards, status} = current;
        const playing = String(current.playing);
        if (current.root.dataset.simulationPlaying !== playing) current.root.dataset.simulationPlaying = playing;
        // Read all moving points before DOM writes to avoid alternating SVG
        // geometry reads and layout-invalidating updates for every edge.
        const edgeStates = current.edges.map(edge => {
            const transfer = edge.transfers.find(value => time >= value.start && time < value.end);
            return {edge, transfer,
                position: transfer ? edge.path.getPointAtLength((time - transfer.start) * packetSpeed) : null};
        });
        cards.forEach(card => {
            const jobs = card.jobs || [];
            const finished = jobs.filter(job => time >= job.end).length;
            const active = jobs.filter(job => time >= job.start && time < job.end);
            const state = !card.enabled ? 'disabled' : !jobs.length ? 'skipped'
                : active.length ? 'processing' : finished === jobs.length ? 'completed' : 'waiting';
            if (card.node.dataset.simulationState !== state) card.node.dataset.simulationState = state;
            const stateText = (state === 'skipped' && card.condition ? '조건부 경로 · 이번 모의 실행에서 제외' : stateLabels[state])
                + (jobs.length > 1 ? ` · ${finished}/${jobs.length}개` : '');
            if (card.state.textContent !== stateText) card.state.textContent = stateText;
            const sample = active[0] || jobs[0];
            const dataText = (sample ? `${sample.input} → ${sample.output}` : card.description).replaceAll('CALL2-SELECT', 'SLOT-SELECTER').replaceAll('CALL2-PLAN', 'PLAN').replaceAll('CALL2-DETAIL', 'DETAIL').replaceAll('CALL2-KEYVIS', 'KEYVIS').replaceAll('CALL3', 'POST PROCESS');
            if (card.data.textContent !== dataText) card.data.textContent = dataText;
        });
        current.groupStatuses.forEach(group => {
            const slots = new Map();
            (group.slotJobs || []).forEach(job => {
                const counts = slots.get(job.llm.slot) || {limit: job.llm.limit, active: 0, waiting: 0, completed: 0, total: 0};
                counts.total++;
                if (time >= job.end) counts.completed++;
                else if (time >= job.start) counts.active++;
                else counts.waiting++;
                slots.set(job.llm.slot, counts);
            });
            const groupText = [...slots].map(([slot, counts]) =>
                `${slot.toUpperCase()} 슬롯 전체 · 상한 ${counts.limit}개 · 처리 중 ${counts.active}개 · 대기 ${counts.waiting}개 · 완료 ${counts.completed}/${counts.total}`
            ).join(' / ') || 'OFF · 실행 대상 없음';
            if (group.node.textContent !== groupText) group.node.textContent = groupText;
        });
        edgeStates.forEach(({edge, transfer, position}) => {
            const display = transfer ? '' : 'none';
            if (edge.packet.style.display !== display) edge.packet.style.display = display;
            const transferred = String(edge.transfers.some(value => time >= value.end));
            if (edge.path.dataset.transferred !== transferred) edge.path.dataset.transferred = transferred;
            if (transfer) {
                const progress = (time - transfer.start) / (transfer.end - transfer.start);
                edge.packet.setAttribute('cx', position.x); edge.packet.setAttribute('cy', position.y);
                edge.packet.style.opacity = String(Math.min(1, progress * 6, (1 - progress) * 6));
            }
        });
        const activeNames = [...new Set(model.jobs.filter(job => time >= job.start && time < job.end).map(job => job.name.replaceAll('CALL2-SELECT', 'SLOT-SELECTER').replaceAll('CALL2-PLAN', 'PLAN').replaceAll('CALL2-DETAIL', 'DETAIL').replaceAll('CALL2-KEYVIS', 'KEYVIS').replaceAll('CALL3', 'POST PROCESS')))];
        const text = model.blocked.length ? '실행 계획에 진행할 수 없는 작업이 있습니다' : activeNames.length
            ? `FLOW 자동 재생 · ${activeNames.join(' + ')}`
            : 'FLOW 자동 재생 · 다음 단계로 데이터 전달 중';
        if (status.textContent !== text) status.textContent = text;
    }

    function pause() {
        if (!current) return;
        current.playing = false;
        cancelAnimationFrame(current.frame);
        current.frame = 0;
        paint();
    }

    function tick(now) {
        if (!current?.playing) return;
        if (now < current.nextFrameAt - 0.1) {
            current.frame = requestAnimationFrame(tick);
            return;
        }
        if (!isVisible()) {pause(); return;}
        current.time = (current.time + (now - current.lastTime) / 1000) % current.model.duration;
        current.lastTime = now;
        // Keep a 60 Hz deadline without slowing the simulation on a high-refresh
        // monitor or replaying missed frames after a stall. Allow clock rounding.
        current.nextFrameAt += frameInterval * Math.max(1, Math.floor((now - current.nextFrameAt) / frameInterval) + 1);
        paint();
        current.frame = requestAnimationFrame(tick);
    }

    function play() {
        if (!current || current.playing || !isVisible()) return;
        if (current.time >= current.model.duration) current.time = 0;
        current.playing = true; current.lastTime = performance.now();
        current.nextFrameAt = current.lastTime + frameInterval;
        paint(); current.frame = requestAnimationFrame(tick);
    }

    function refresh(root, plan) {
        const container = root.closest('.illust-flow-preview');
        const toolbar = container.querySelector('.illust-preview-toolbar');
        const cards = new Map([...root.querySelectorAll('[data-illust-preview-node]')].map(node => {
            const name = node.dataset.illustPreviewNode;
            const id = node.dataset.previewId;
            const description = node.querySelector('span').textContent;
            node.title = description;
            node.classList.add('has-simulation');
            const state = element('span', 'illust-preview-state');
            const data = element('span', 'illust-preview-data');
            node.append(state, data);
            return [id, {id, name, node, state, data, description,
                condition: node.dataset.condition, enabled: node.dataset.enabled === 'true'}];
        }));
        const signature = JSON.stringify(plan);
        const previous = current;
        if (previous) {
            cancelAnimationFrame(previous.frame); previous.observer.disconnect(); previous.visibilityObserver.disconnect();
        }
        const svg = svgElement('svg', {class: 'illust-preview-connections', 'aria-hidden': 'true'});
        root.querySelector('.illust-preview-connections')?.remove(); root.append(svg);
        const model = buildIllustrationPreviewSimulation(plan);
        const groupStatuses = [...root.querySelectorAll('[data-illust-preview-group]')].map(group => {
            const cardIds = new Set([...group.querySelectorAll('[data-illust-preview-node]')]
                .map(node => node.dataset.previewId));
            const summary = element('p', 'illust-preview-group-state');
            summary.setAttribute('role', 'status');
            group.insertBefore(summary, group.children[1]);
            return {node: summary, jobs: model.jobs.filter(job => job.llm && cardIds.has(job.card))};
        });
        current = {root, cards, plan, model, connections: buildIllustrationPreviewConnections(model, plan), svg, edges: [], frame: 0, signature,
            groupStatuses,
            time: previous?.signature === signature ? previous.time : 0,
            playing: false,
            status: toolbar.querySelector('[data-illust-preview-status]'),
            observer: new ResizeObserver(drawConnections), visibilityObserver: new MutationObserver(drawConnections)};
        current.observer.observe(root);
        for (let ancestor = root.parentElement; ancestor; ancestor = ancestor.parentElement) {
            current.visibilityObserver.observe(ancestor, {attributes: true, attributeFilter: ['hidden', 'class', 'style']});
        }
        drawConnections();
    }

    document.addEventListener('visibilitychange', () => document.hidden ? pause() : play());
    window.getIllustrationPreviewLLM = getIllustrationPreviewLLM;
    window.refreshIllustrationNodePreviewAnimation = refresh;
    window.renderIllustrationPreviewPlan = renderIllustrationPreviewPlan;
    window.pauseIllustrationPreview = pause;
    window.getIllustrationPreviewTasks = names => [...new Set(names.map(name => taskKeys[name]
        || taskKeys[Object.keys(taskKeys).find(key => name.startsWith(key + ' '))]).filter(Boolean))];
})();
