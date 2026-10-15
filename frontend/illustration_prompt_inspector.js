/* Show the current shared assembly using named placeholders. */
(() => {
    let root, status, content, stages = [], selectedKey = 'select', refreshVersion = 0;
    const roles = {
        system: ['시스템 프롬프트 (SYSTEM)', 'LLM이 따라야 할 작업 지침·판단 기준·출력 규칙'],
        user: ['유저 프롬프트 (USER)', '서버가 변수와 코드 규칙을 조립해서 전달하는 처리 대상'],
        assistant: ['이전 대화 (ASSISTANT)', '일반 대화 경로에서 전달하는 이전 캐릭터 발화'],
        positive: ['긍정 이미지 프롬프트', '출력 템플릿에 이미지 태그·대사·추가 지침을 삽입한 결과'],
        negative: ['부정 이미지 프롬프트', '고정 부정 태그와 캐릭터별·추가 부정 태그의 조합'],
    };
    const el = (tag, className, text) => {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    };
    function token(label) { return el('mark', 'illust-input-variable', '{' + label + '}'); }
    function promptText(pre, text, spans) {
        let cursor = 0;
        for (const span of spans || []) {
            if (!Number.isInteger(span.start) || !Number.isInteger(span.end) || span.start < cursor || span.end <= span.start || span.end > text.length) {
                console.error('[ILLUST_STRUCTURE] 변수 위치 오류:', span, text.length);
                continue;
            }
            pre.append(document.createTextNode(text.slice(cursor, span.start)));
            const mark = el('mark', 'illust-input-variable', text.slice(span.start, span.end));
            mark.title = '변수: ' + span.label;
            pre.append(mark); cursor = span.end;
        }
        pre.append(document.createTextNode(text.slice(cursor)));
    }
    function recipe(parts) {
        const row = el('div', 'illust-structure-recipe');
        parts.forEach((part, index) => {
            const label = typeof part === 'string' ? part : part.label;
            if (index) row.append(el('span', 'illust-structure-plus', '+'));
            const item = label.startsWith('{') && label.endsWith('}') ? token(label.slice(1, -1)) : el('span', 'illust-structure-fixed', label);
            if (part.source) item.title = part.source;
            row.append(item);
        });
        return row;
    }
    function assemblyBlocks(stage) {
        const container = el('section', 'illust-structure-blocks');
        container.append(el('h4', '', '조립 순서 · 블록별 출처와 내용'),
            el('p', 'illust-structure-hint', '같은 역할의 블록은 위에서 아래 순서로 연결합니다. 블록을 펼치면 조립에 쓰이는 내용을 확인할 수 있습니다.'));
        let count = 0;
        for (const [role, parts] of [['SYSTEM', stage.system_parts], ['USER', stage.user_parts || []]]) {
            parts.forEach((part, index) => {
                if (typeof part === 'string') return;
                count++;
                const block = el('details', 'illust-structure-block');
                block.dataset.role = role; block.dataset.source = part.path || part.source;
                const summary = el('summary');
                summary.append(el('span', 'illust-structure-block-order', `${role} ${index + 1}`),
                    recipe([part]));
                const source = el('p', 'illust-structure-block-source', '출처: ' + part.source);
                if (part.path) source.title = part.path;
                block.append(summary, source);
                if (part.condition) block.append(el('p', 'illust-structure-condition', '포함 조건: ' + part.condition));
                const pre = el('pre'); pre.tabIndex = 0;
                promptText(pre, part.content, part.input_variables); block.append(pre);
                container.append(block);
            });
        }
        return count ? container : null;
    }
    function renderMessages(container, data) {
        const messages = data.messages || [];
        const variables = data.input_variables || [];
        container.replaceChildren();
        const grid = el('div', 'illust-structure-messages');
        messages.forEach((message, index) => {
            const [title, help] = roles[message.role] || [message.role, '조립된 메시지'];
            const card = el('section', 'illust-input-message'); card.dataset.role = message.role;
            card.append(el('h4', '', title), el('p', 'illust-input-role-help', help));
            const pre = el('pre'); pre.tabIndex = 0;
            promptText(pre, message.content, variables[index]); card.append(pre); grid.append(card);
        });
        container.append(grid);
    }
    function selectStage(key, variantKey = '') {
        const definition = stages.find(item => item.key === key);
        if (!definition) { console.error('[ILLUST_STRUCTURE] 단계 없음:', key); return; }
        const variant = definition.variants?.find(item => item.key === variantKey);
        const stage = {...definition, ...variant, key: definition.key};
        selectedKey = key;
        root.querySelectorAll('.illust-structure-node').forEach(node => {
            node.classList.toggle('active', node.dataset.key === key);
            node.setAttribute('aria-pressed', String(node.dataset.key === key));
        });
        const detail = root.querySelector('#illust-structure-detail');
        detail.replaceChildren(el('h3', '', stage.title), el('p', 'illust-structure-purpose', stage.purpose));
        if (stage.condition) detail.append(el('p', 'illust-structure-condition', '실행 조건: ' + stage.condition));
        const formula = el('div', 'illust-structure-formula');
        formula.append(el('strong', '', stage.key === 'assemble' ? '출력 조립' : '시스템 조립'), recipe(stage.system_parts));
        if (stage.user_parts) formula.append(el('strong', '', '유저 조립에 포함'), recipe(stage.user_parts));
        const userLabels = [...new Set(stage.input_variables.flatMap((spans, index) =>
            stage.messages[index].role === 'system' ? [] : spans.map(span => span.label)))];
        formula.append(el('strong', '', stage.key === 'assemble' ? '삽입 변수' : '유저 입력 변수'));
        const inputs = el('div', 'illust-structure-recipe');
        userLabels.forEach(label => inputs.append(token(label)));
        formula.append(inputs); detail.append(formula);
        const blocks = assemblyBlocks(stage);
        if (blocks) detail.append(blocks);
        const output = el('div', 'illust-structure-output'); output.append(el('strong', '', '결과 → '));
        stage.outputs.forEach(label => output.append(token(label)));
        const messages = el('div');
        if (stage.variants) {
            const label = el('label', 'illust-structure-variant', '지침 모드');
            const select = el('select'); select.setAttribute('aria-label', stage.key === 'call3' ? '대사 프롬프트 모드' : '마스크 예측 방식');
            stage.variants.forEach(variant => {const option = el('option', '', variant.label); option.value = variant.key; select.append(option);});
            select.value = variantKey || stage.variants[0].key;
            select.onchange = () => selectStage(key, select.value);
            label.append(select); detail.append(label);
        }
        detail.append(messages); renderMessages(messages, stage);
        detail.append(output);
        if (stage.notes) detail.append(el('p', 'illust-structure-notes', stage.notes));
    }
    function render(data) {
        stages = data.stages;
        content.replaceChildren();
        const overview = el('section', 'illust-structure-overview');
        overview.append(el('h3', '', '전체 조립 흐름'), el('p', 'illust-structure-hint', '단계를 누르면 시스템·유저 프롬프트의 전체 구성을 아래에서 확인할 수 있습니다.'));
        const flow = el('div', 'illust-structure-flow');
        flow.style.setProperty('--illust-structure-rows', String(data.flow.reduce((rows, group) => Math.max(rows, group.keys.length + 1), 1)));
        data.flow.forEach((group, index) => {
            if (index) overview.append(el('div', 'illust-structure-arrow', '↓'));
            const lane = el('div', 'illust-structure-lane');
            lane.append(el('p', 'illust-structure-lane-label', `${index + 1}. ${group.label}`));
            const nodes = el('div', 'illust-structure-nodes');
            group.keys.forEach(key => {
                const stage = stages.find(item => item.key === key);
                const node = el('button', 'illust-structure-node'); node.type = 'button'; node.dataset.key = key;
                node.setAttribute('aria-controls', 'illust-structure-detail');
                node.append(el('strong', '', stage.title));
                if (stage.condition) node.append(el('span', 'illust-structure-condition', stage.condition));
                node.title = '결과: ' + stage.outputs.map(label => '{' + label + '}').join(' · ');
                node.onclick = () => selectStage(key); nodes.append(node);
            });
            lane.append(nodes); flow.append(lane);
        });
        overview.append(flow);
        const detail = el('section', 'illust-structure-detail'); detail.id = 'illust-structure-detail';
        content.append(overview, detail);
        selectStage(stages.some(stage => stage.key === selectedKey) ? selectedKey : stages[0].key);
        status.textContent = '현재 조립 구조 · ' + stages.length + '단계 · 조건부 입력도 함께 표시';
    }
    async function refresh() {
        if (!root) return;
        const version = ++refreshVersion;
        status.textContent = '현재 프롬프트 구성을 불러오는 중…';
        try {
            const response = await fetch('/api/illustration_context/prompt-structure', {cache: 'no-store'});
            if (!response.ok) throw Error('구성 조회 실패 (HTTP ' + response.status + ')');
            const data = await response.json();
            if (version !== refreshVersion) return;
            if (!Array.isArray(data.stages) || !data.stages.length || !Array.isArray(data.flow)) throw Error('표시할 조립 구성이 없습니다.');
            render(data);
        } catch (error) {
            console.error('[ILLUST_STRUCTURE] 구성 조회 실패:', error);
            if (version !== refreshVersion) return;
            status.textContent = error.message;
            content.replaceChildren(el('p', 'illust-input-error', error.message + ' · 구성 새로고침으로 다시 불러올 수 있습니다.'));
        }
    }
    function mount() {
        root = document.getElementById('illust-assembled-inputs');
        if (!root || root.dataset.mounted) return;
        root.dataset.mounted = 'true';
        const style = el('style');
        style.textContent = `
            #illust-assembled-inputs{line-height:1.55;min-width:0}
            #illust-assembled-inputs h3{font-size:.95rem;margin:0 0 8px}
            #illust-assembled-inputs .illust-input-variable{background:#174678;color:#edf6ff;border-radius:4px;padding:2px 5px;box-decoration-break:clone;-webkit-box-decoration-break:clone;overflow-wrap:anywhere}
            #illust-assembled-inputs .illust-structure-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:.8rem;margin:12px 0}
            #illust-assembled-inputs .illust-structure-fixed{border:1px solid var(--border);background:var(--surface2);padding:2px 6px;border-radius:4px}
            #illust-assembled-inputs .illust-structure-hint,#illust-assembled-inputs .illust-structure-status{font-size:.78rem;color:var(--text2);margin:6px 0}
            #illust-assembled-inputs .illust-structure-overview{padding:14px;border:1px solid var(--border);border-radius:8px}
            #illust-assembled-inputs .illust-structure-flow{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px 16px;margin-top:12px}
            #illust-assembled-inputs .illust-structure-lane{position:relative;min-width:0;display:grid;grid-template-columns:minmax(0,1fr);grid-template-rows:subgrid;grid-row:span var(--illust-structure-rows)}
            #illust-assembled-inputs .illust-structure-lane:not(:first-child)::before{content:'→';position:absolute;grid-column:1 / 2;grid-row:2 / 3;left:-14px;top:50%;transform:translateY(-50%);color:var(--text2);font-size:12px}
            #illust-assembled-inputs .illust-structure-lane-label{font-size:.72rem;color:var(--text2);margin:0}
            #illust-assembled-inputs .illust-structure-nodes{display:grid;grid-template-rows:subgrid;grid-row:2 / -1;min-width:0}
            #illust-assembled-inputs .illust-structure-node{width:100%;min-width:0;display:flex;flex-direction:column;gap:4px;text-align:left;border:1px solid var(--border);border-radius:6px;padding:9px 10px;color:var(--text);background:var(--surface);cursor:pointer;font-size:.76rem;overflow-wrap:anywhere}
            #illust-assembled-inputs .illust-structure-node.active{border-color:#81baff;box-shadow:inset 3px 0 #81baff}
            #illust-assembled-inputs .illust-structure-node:focus-visible{outline:2px solid #81baff;outline-offset:2px}
            #illust-assembled-inputs .illust-structure-node-result{font-size:.72rem;color:#a8ceff}
            #illust-assembled-inputs .illust-structure-condition{font-size:.72rem;color:var(--text2);margin:0}
            #illust-assembled-inputs .illust-structure-arrow{display:none}
            #illust-assembled-inputs .illust-structure-detail{margin-top:18px}
            #illust-assembled-inputs .illust-structure-purpose{margin:4px 0 8px;font-size:.82rem}
            #illust-assembled-inputs .illust-structure-formula{display:grid;grid-template-columns:95px 1fr;gap:9px;font-size:.8rem;align-items:start;padding:12px 0}
            #illust-assembled-inputs .illust-structure-recipe,#illust-assembled-inputs .illust-structure-output{display:flex;gap:5px 7px;flex-wrap:wrap;align-items:center;min-width:0}
            #illust-assembled-inputs .illust-structure-output{font-size:.8rem}
            #illust-assembled-inputs .illust-structure-plus{color:var(--text2)}
            #illust-assembled-inputs .illust-structure-blocks{margin:6px 0 16px;border:1px solid var(--border);border-radius:7px;overflow:hidden}
            #illust-assembled-inputs .illust-structure-blocks>.illust-structure-hint{padding:0 12px}
            #illust-assembled-inputs .illust-structure-block{border-top:1px solid var(--border);font-size:.78rem}
            #illust-assembled-inputs .illust-structure-block summary{display:flex;align-items:center;gap:8px;padding:9px 12px;cursor:pointer;overflow-wrap:anywhere}
            #illust-assembled-inputs .illust-structure-block summary::before{content:'▸';color:var(--text2)}
            #illust-assembled-inputs .illust-structure-block[open] summary::before{content:'▾'}
            #illust-assembled-inputs .illust-structure-block-order{font-size:.7rem;color:var(--text2);white-space:nowrap}
            #illust-assembled-inputs .illust-structure-block-source{padding:0 12px;margin:4px 0;overflow-wrap:anywhere}
            #illust-assembled-inputs .illust-structure-block>.illust-structure-condition{padding:0 12px;margin:4px 0 8px}
            #illust-assembled-inputs .illust-structure-notes{font-size:.78rem;color:var(--text2);padding:9px 12px;border-left:2px solid var(--border);margin:12px 0}
            #illust-assembled-inputs .illust-structure-messages{display:grid;grid-template-columns:1fr 1fr;gap:12px}
            #illust-assembled-inputs .illust-input-message{min-width:0;border:1px solid var(--border);border-radius:7px;overflow:hidden}
            #illust-assembled-inputs h4{font-size:.85rem;margin:0;padding:10px 12px;background:var(--surface2)}
            #illust-assembled-inputs .illust-input-role-help{margin:0;padding:7px 12px;font-size:.75rem;color:var(--text2)}
            #illust-assembled-inputs pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,monospace;max-height:430px;overflow:auto;padding:12px;margin:0;background:var(--surface);color:var(--text)}
            #illust-assembled-inputs .illust-structure-variant{display:flex;align-items:center;gap:10px;margin:12px 0;font-size:.8rem}
            #illust-assembled-inputs select{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:6px}
            #illust-assembled-inputs .illust-input-error{color:#ff8585;overflow-wrap:anywhere}
            @media(max-width:1000px){#illust-assembled-inputs .illust-structure-flow{grid-template-columns:repeat(4,minmax(0,1fr))}#illust-assembled-inputs .illust-structure-lane::before{display:none}}
            @media(max-width:650px){#illust-assembled-inputs .illust-structure-flow{grid-template-columns:repeat(2,minmax(0,1fr))}#illust-assembled-inputs .illust-structure-messages{grid-template-columns:1fr}#illust-assembled-inputs .illust-structure-formula{grid-template-columns:1fr}}
        `;
        root.append(style, el('p', '', '입력은 {변수명}으로 표시하고, 현재 지침과 코드가 프롬프트를 조립하는 순서를 보여줍니다.'));
        const legend = el('div', 'illust-structure-legend');
        legend.append(token('과거 문맥 입력'), el('span', '', '파란 배경: 바뀌는 입력·설정'), el('span', 'illust-structure-fixed', '기본 배경: 고정 지침·코드 규칙'));
        root.append(legend);
        status = el('p', 'illust-structure-status'); status.setAttribute('aria-live', 'polite');
        content = el('div'); root.append(status, content);
    }
    window.mountIllustrationPromptInspector = mount;
    window.refreshIllustrationPromptInspector = refresh;
})();
