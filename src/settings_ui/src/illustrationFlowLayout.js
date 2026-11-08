// Stage and branch layout from frontend/illustration_flow.js (the original workflow viewer).
export function layoutNodes(nodes, kind) {
    const isIllustration = kind === 'illustration';
    const imageDepth = nodes.some(n => n.kind === 'reference') ? 7 : 6;
    if (kind === 'illustration' && nodes.some(n => ['illustration_easy_edit', 'regenerate'].includes(n.task_key))) kind = 'illustration_edit';
    // Display stages are fixed even when branches finish or dispatch images early.
    // Names also classify reserved LLM nodes before routing metadata arrives.
    const stageRules = [
        ['SLOT-SELECTER', 'call2_select', 2, 3],
        ['MANGA-CHARACTER-RESOLVE', 'manga_select', 1, 0],
        ['PLANNER-MANGA', 'manga_plan', 3, 0],
        ['DETAIL-MANGA', 'manga_detail', 4, 0],
        ['PLAN', 'call2_plan', 3, 0],
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
        ['NPC 생성 노드', 'npc_generation', 2, 3],
        ['NPC 대표 이미지', 'npc_generation_image', 2, 4],
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
        ['레퍼런스 전처리', 'reference_preprocessing', 6, 0],
        ['SLOT', 'illustration_images', imageDepth, 0],
        ['이미지 생성', 'illustration_generation', imageDepth, 1],
    ];
    const stageOf = n => {
        if (kind !== 'illustration' || n.kind === 'request' || n.kind === 'result') return null;
        // Each correction/redraw follows its own dependency; grouping it
        // with the first generation would turn the displayed loop into a cycle.
        if (String(n.layout_group || '').startsWith('visual_correction_')) return null;
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
    const layoutGroup = n => kind === 'illustration_edit' ? n.id : stageById.get(n.id)
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
        if (members.some(n => n.kind === 'request') && !dependenciesByGroup.get(group)?.size) {depthByGroup.set(group, 0); return 0;}
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
    const top = isIllustration ? 104 : 32;
    const depthById = new Map(nodes.map(n => [n.id, groupDepth(layoutGroup(n))]));
    const laneById = new Map();
    if (kind === 'illustration') {
        const plans = orderedNodes.filter(n => ['call2_plan', 'manga_plan'].includes(stageById.get(n.id)?.[1]));
        const detailsByPlan = new Map(plans.map(n => [n.id, []]));
        const planAncestors = (id, visiting = new Set()) => {
            if (!nodeById.has(id) || visiting.has(id)) return new Set();
            if (detailsByPlan.has(id)) return new Set([id]);
            const nextVisiting = new Set(visiting); nextVisiting.add(id);
            return new Set((nodeById.get(id).dependencies || [])
                .flatMap(parent => [...planAncestors(parent, nextVisiting)]));
        };
        orderedNodes.filter(n => ['call2_detail', 'manga_detail'].includes(stageById.get(n.id)?.[1])).forEach(n => {
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
        const cardHeight = isIllustration && nodePreviewKind(n) ? 360 : 98;
        rowHeights.set(key, Math.max(rowHeights.get(key) || 0, cardHeight));
    });
    positions.forEach(p => {
        const group = kind === 'illustration' && p.depth >= 5 ? 'branches' : p.depth;
        let y = top;
        for (let row = 0; row < p.y; row += 1) y += (rowHeights.get(`${group}:${row}`) || 98) + 38;
        p.y = y;
    });
    const width = Math.max(isIllustration ? rootColumnX + 7 * 290 + 270 : 650, ...[...positions.values()].map(p => p.x + 270));
    const height = Math.max(350, ...nodes.map(n => positions.get(n.id).y + (isIllustration && nodePreviewKind(n) ? 360 : 98) + 42));
    return {positions, width, height};
}
export function nodePreviewKind(n) {
    const caption = String(n.call_name || n.label || '');
    if (n.task_key === 'illustration_multi_char_mask' || n.layout_group === 'illustration_multi_char_mask'
            || caption === 'MULTI-CHAR-MASK' || caption.startsWith('MULTI-CHAR-MASK ')
            || caption === 'MULTI-CHAR_MASK' || caption.startsWith('MULTI-CHAR_MASK ')) return 'mask';
    return n.kind === 'image' || n.kind === 'reference' || n.task_key === 'illustration_visual_selection'
        || n.layout_group === 'illustration_generation' || n.image_nodes?.length ? 'image' : '';
}
export function illustrationSlot(n) {
    const caption = String(n.call_name || n.label || '');
    const slot = n.illustration_slot ?? (n.inspection_images?.length === 1 ? n.inspection_images[0].slot : undefined)
        ?? caption.match(/\bslot(?:\s*=\s*|\s+)(-?\d+)\b/)?.[1];
    return slot !== undefined && slot !== null && slot !== '' && Number.isInteger(Number(slot)) ? Number(slot) : null;
}

export function imageMaskNode(n, flow) {
    const masks = flow.nodes.filter(node => nodePreviewKind(node) === 'mask');
    const slot = illustrationSlot(n);
    const panel = String(n.call_name || n.label || '').match(/컷\s*(\d+)/)?.[1];
    const matching = slot === null ? [] : masks.filter(mask => illustrationSlot(mask) === slot
        && (!panel || String(mask.call_name || mask.label || '').match(/컷\s*(\d+)/)?.[1] === panel));
    const dependencies = masks.filter(mask => n.dependencies?.includes(mask.id));
    const candidates = matching.length ? matching
        : dependencies.length === 1 && (slot === null || illustrationSlot(dependencies[0]) === null) ? dependencies : [];
    return [...candidates].reverse().find(mask => mask.status === 'completed') || candidates.at(-1);
}
