/* Settings-only explanation of the illustration pipeline. No generation requests. */
(() => {
    'use strict';
    const transferTime = 0.08;
    const playbackDuration = {llm: 1.6, render: 3.2, local: 0.08};

    const taskKeys = {
        'CHARACTER-RESOLVE': 'illustration_character_resolve',
        'PROFILE-CONTEXT-TRANSLATE': 'illustration_profile_context_translate',
        'PROFILE-RESOLVE': 'illustration_profile_resolve',
        'CALL1-BACKTRANSLATE': 'illustration_call1_backtranslate',
        'CALL1': 'illustration_call1',
        'CALL2-SELECT': 'illustration_call2_select',
        'CALL2-PLAN': 'illustration_call2_plan',
        'CALL2-DETAIL': 'illustration_call2',
        'CALL2-KEYVIS': 'illustration_call2_keyvis',
        'CALL3': 'illustration_call3',
        'MULTI-CHAR-MASK': 'illustration_multi_char_mask',
        'ORIGINAL-ASSET': 'illustration_original_asset',
        'ORIGINAL-ASSET-RECOVERY': 'illustration_original_asset_recovery',
        'IMAGE-QUALITY-INSPECTION': 'illustration_quality_inspection',
        'SET-QUALITY-INSPECTION': 'illustration_quality_inspection',
    };

    function getIllustrationPreviewLLM(name, config = {}) {
        const task = taskKeys[name];
        if (!task) return null;
        let slot = config.llm_routing?.[task]?.primary ?? 'llm1';
        if (!/^llm(?:[1-9]|10)$/.test(slot)) {
            console.error('[ILLUST_PREVIEW] LLM 슬롯이 올바르지 않아 LLM1을 사용합니다:', {task, slot});
            slot = 'llm1';
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

    function buildIllustrationPreviewSimulation(plan) {
        // Relative playback durations distinguish external work from local steps.
        // Dependencies and resource limits still come from the execution plan.
        const jobs = plan.jobs.map(job => ({...job, parents: [...job.parents], start: Infinity, end: Infinity}));
        const byId = new Map(jobs.map(job => [job.id, job]));
        const pending = new Set(jobs);
        let active = [], time = 0;
        const readyAt = job => {
            const required = job.parents.map(id => byId.get(id).end);
            if (job.any_parents?.length) required.push(Math.min(...job.any_parents.map(id => byId.get(id).end)));
            return required.length ? Math.max(...required) + transferTime : 0;
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
                job.end = time + (job.llm ? playbackDuration.llm
                    : resources.length ? playbackDuration.render : playbackDuration.local);
                if (job.any_parents?.length) {
                    const first = job.any_parents.reduce((a, b) => byId.get(a).end <= byId.get(b).end ? a : b);
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

    function buildIllustrationPreviewConnections(model, plan) {
        const byId = new Map(model.jobs.map(job => [job.id, job]));
        const transfers = model.jobs.flatMap(job => (job.data_parents ?? job.parents).map(parentId =>
            ({from: byId.get(parentId).card, to: job.card,
                start: byId.get(parentId).end, end: byId.get(parentId).end + transferTime,
                label: byId.get(parentId).output})));
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
        const labels = {backtranslation: '역번역 배치', call1_select: 'CALL1 + SELECT',
            call2_plan: 'PLAN 배치', call2_detail: 'DETAIL 배치', keyvis: 'KEYVIS',
            original_asset: 'Original Asset'};
        const nodeHTML = node => `<article class="illust-flow-preview-node${node.enabled ? '' : ' is-disabled'}${node.group === activeGroup ? ' is-current' : ''}"
            data-illust-preview-node="${escape(node.name)}" data-preview-id="${escape(node.id)}"
            data-enabled="${node.enabled}" data-condition="${escape(node.condition)}"
            data-batch-index="${node.index}" data-batch-count="${node.count}">
            <strong>${escape(node.name)}${node.count > 1 ? `<small class="illust-flow-preview-batch-index">${node.index}/${node.count}</small>` : ''}<small class="illust-preview-activation" data-activation="${node.enabled ? 'on' : 'off'}">사용 ${node.enabled ? 'ON' : 'OFF'}</small></strong>
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
            return `<section class="illust-flow-preview-step" aria-label="STEP ${index + 1}"><h4 class="illust-flow-preview-step-title">STEP ${index + 1} · ${escape(title)}</h4>${content}</section>`;
        }).join('');
        const container = root.closest('.illust-flow-preview');
        container.querySelector('[data-illust-preview-summary]').textContent = plan.summary;
        container.querySelector('.illust-flow-preview-note').textContent = plan.notes.join(' ');
        refresh(root, plan);
    }

    let current = null;
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
        && current.root.closest('.modal')?.classList.contains('visible')
        && !current.root.closest('#illust-tab-content-toggles')?.hidden;

    function drawConnections() {
        if (!current) return;
        const {root, cards, model, svg} = current;
        const bounds = root.getBoundingClientRect();
        svg.setAttribute('width', root.clientWidth);
        svg.setAttribute('height', root.clientHeight);
        svg.replaceChildren();
        const marker = svgElement('marker', {id: 'illust-preview-arrow', viewBox: '0 0 10 10',
            refX: '8', refY: '5', markerWidth: '5', markerHeight: '5', orient: 'auto-start-reverse'});
        marker.append(svgElement('path', {d: 'M 0 0 L 10 5 L 0 10 z', fill: 'currentColor'}));
        const defs = svgElement('defs', {}); defs.append(marker); svg.append(defs);
        current.edges = [];
        current.connections.forEach(({from: parent, to: job, transfers}) => {
            const a = cards.get(parent.id).node.getBoundingClientRect();
            const b = cards.get(job.id).node.getBoundingClientRect();
            const ax = a.right - bounds.left, ay = a.top + a.height / 2 - bounds.top;
            const bx = b.left - bounds.left, by = b.top + b.height / 2 - bounds.top;
            const sameColumn = Math.abs(a.left - b.left) < 40;
            let d;
            if (sameColumn) {
                const end = b.right - bounds.left, lane = Math.max(ax, end) + 6;
                d = `M ${ax} ${ay} C ${lane} ${ay}, ${lane} ${by}, ${end} ${by}`;
            } else if (bx - ax > 210 || bx < ax) {
                d = `M ${ax} ${ay} H ${ax + 5} V 20 H ${bx - 5} V ${by} H ${bx}`;
            } else {
                const middle = (ax + bx) / 2;
                d = `M ${ax} ${ay} C ${middle} ${ay}, ${middle} ${by}, ${bx} ${by}`;
            }
            const path = svgElement('path', {d, class: 'illust-preview-edge', 'marker-end': 'url(#illust-preview-arrow)',
                'data-from-node': parent.id, 'data-to-node': job.id, 'data-simulated': String(transfers.length > 0)});
            const packet = svgElement('circle', {r: '4', class: 'illust-preview-packet'});
            const label = svgElement('text', {class: 'illust-preview-packet-label'});
            svg.append(path, packet, label);
            current.edges.push({path, packet, label, transfers});
        });
        paint();
    }

    function paint() {
        if (!current) return;
        const {time, model, cards, play, progress, status} = current;
        current.root.dataset.simulationPlaying = String(current.playing);
        const grouped = new Map();
        model.jobs.forEach(job => {
            const list = grouped.get(job.card) || [];
            list.push(job); grouped.set(job.card, list);
        });
        cards.forEach(card => {
            const jobs = grouped.get(card.id) || [];
            const finished = jobs.filter(job => time >= job.end).length;
            const active = jobs.filter(job => time >= job.start && time < job.end);
            const state = !card.enabled ? 'disabled' : !jobs.length ? 'skipped'
                : active.length ? 'processing' : finished === jobs.length ? 'completed' : 'waiting';
            card.node.dataset.simulationState = state;
            card.state.textContent = (state === 'skipped' && card.condition ? '조건부 경로 · 이번 모의 실행에서 제외' : stateLabels[state])
                + (jobs.length > 1 ? ` · ${finished}/${jobs.length}개` : '');
            const sample = active[0] || jobs[0];
            card.data.textContent = sample ? `${sample.input} → ${sample.output}` : card.description;
            const amount = jobs.length ? jobs.reduce((sum, job) => sum + (Number.isFinite(job.start)
                ? Math.max(0, Math.min(1, (time - job.start) / (job.end - job.start))) : 0), 0) / jobs.length : 0;
            card.bar.style.width = `${amount * 100}%`;
        });
        current.groupStatuses.forEach(group => {
            const slots = new Map();
            const groupSlots = new Set(group.jobs.map(job => job.llm.slot));
            model.jobs.filter(job => job.llm && groupSlots.has(job.llm.slot)).forEach(job => {
                const counts = slots.get(job.llm.slot) || {limit: job.llm.limit, active: 0, waiting: 0, completed: 0, total: 0};
                counts.total++;
                if (time >= job.end) counts.completed++;
                else if (time >= job.start) counts.active++;
                else counts.waiting++;
                slots.set(job.llm.slot, counts);
            });
            group.node.textContent = [...slots].map(([slot, counts]) =>
                `${slot.toUpperCase()} 슬롯 전체 · 상한 ${counts.limit}개 · 처리 중 ${counts.active}개 · 대기 ${counts.waiting}개 · 완료 ${counts.completed}/${counts.total}`
            ).join(' / ') || 'OFF · 실행 대상 없음';
        });
        current.edges.forEach(edge => {
            const transfer = edge.transfers.find(value => time >= value.start && time < value.end);
            const moving = Boolean(transfer);
            edge.packet.style.display = edge.label.style.display = moving ? '' : 'none';
            edge.path.dataset.transferred = String(edge.transfers.some(value => time >= value.end));
            if (moving) {
                edge.label.textContent = transfer.label;
                const position = edge.path.getPointAtLength(edge.path.getTotalLength() * (time - transfer.start) / transferTime);
                edge.packet.setAttribute('cx', position.x); edge.packet.setAttribute('cy', position.y);
                edge.label.setAttribute('x', position.x + 7); edge.label.setAttribute('y', position.y - 7);
            }
        });
        progress.value = String(time);
        play.textContent = current.playing ? '일시정지' : time >= model.duration ? '다시 재생' : '흐름 재생';
        const activeNames = [...new Set(model.jobs.filter(job => time >= job.start && time < job.end).map(job => job.name))];
        const text = time >= model.duration ? (model.blocked.length ? '실행 자원을 기다리는 작업이 있습니다' : '설정 기반 모의 실행 완료') : activeNames.length
            ? `${current.playing ? '처리 예시' : '멈춤'} · ${activeNames.join(' + ')}`
            : time ? '다음 단계로 데이터 전달 중' : '설정 기반 모의 실행 · LLM·렌더링은 길게, 내부 처리는 빠르게';
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
        if (!isVisible()) {pause(); return;}
        current.time = Math.min(current.model.duration, current.time + (now - current.lastTime) / 1000 * current.speed);
        current.lastTime = now;
        if (current.time >= current.model.duration) current.playing = false;
        paint();
        current.frame = current.playing ? requestAnimationFrame(tick) : 0;
    }

    function play() {
        if (!current || !isVisible()) return;
        if (current.time >= current.model.duration) current.time = 0;
        current.playing = true; current.lastTime = performance.now();
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
            const track = element('div', 'illust-preview-progress');
            const bar = element('i', ''); track.append(bar);
            node.append(state, data, track);
            return [id, {id, name, node, state, data, bar, description,
                condition: node.dataset.condition, enabled: node.dataset.enabled === 'true'}];
        }));
        const signature = JSON.stringify(plan);
        const previous = current;
        if (previous) {
            cancelAnimationFrame(previous.frame); previous.observer.disconnect();
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
        current = {root, cards, model, connections: buildIllustrationPreviewConnections(model, plan), svg, edges: [], frame: 0, signature,
            groupStatuses,
            time: previous?.signature === signature ? previous.time : 0,
            playing: false, speed: previous?.speed || 1,
            play: toolbar.querySelector('[data-illust-preview-play]'),
            progress: toolbar.querySelector('[data-illust-preview-progress]'),
            status: toolbar.querySelector('[data-illust-preview-status]'),
            observer: new ResizeObserver(drawConnections)};
        current.progress.max = String(model.duration);
        current.play.onclick = () => current.playing ? pause() : play();
        toolbar.querySelector('[data-illust-preview-reset]').onclick = () => {pause(); current.time = 0; paint();};
        current.progress.oninput = () => {const time = Number(current.progress.value); pause(); current.time = time; paint();};
        const speed = toolbar.querySelector('[data-illust-preview-speed]');
        speed.value = String(current.speed);
        speed.onchange = () => {current.speed = Number(speed.value);};
        current.observer.observe(root);
        drawConnections();
        if (previous?.signature === signature && previous.playing) play();
    }

    document.addEventListener('visibilitychange', () => {if (document.hidden) pause();});
    window.getIllustrationPreviewLLM = getIllustrationPreviewLLM;
    window.refreshIllustrationNodePreviewAnimation = refresh;
    window.renderIllustrationPreviewPlan = renderIllustrationPreviewPlan;
    window.pauseIllustrationPreview = pause;
    window.getIllustrationPreviewTasks = names => [...new Set(names.map(name => taskKeys[name]).filter(Boolean))];
})();
