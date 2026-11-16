// Shared by the legacy page and the settings bundle. Keep filename rules and
// import ordering identical in both interfaces.
globalThis.ComfyBotBatch = (() => {
    const allowedText = '영문 대소문자, 숫자, 밑줄(_), 하이픈(-), 마침표(.)';
    const reservedNames = new Set(['CON', 'PRN', 'AUX', 'NUL',
        ...Array.from({length: 9}, (_, i) => `COM${i + 1}`),
        ...Array.from({length: 9}, (_, i) => `LPT${i + 1}`)]);

    function getStorageNameError(value, label = '이름') {
        if (typeof value !== 'string') return `${label}은(는) 문자열이어야 합니다.`;
        if (!value) return `${label}을(를) 입력하세요.`;
        if (!/^[A-Za-z0-9_.-]+$/.test(value)) {
            const invalid = [...new Set(Array.from(value).filter(ch => !/[A-Za-z0-9_.-]/.test(ch)))];
            const descriptions = invalid.map(ch => ch === ' ' ? '공백' : JSON.stringify(ch));
            return `${label}에 사용할 수 없는 문자가 있습니다: ${descriptions.join(', ') || '허용되지 않은 문자'}. 사용 가능: ${allowedText}.`;
        }
        if (value === '.' || value === '..' || value.endsWith('.')) return `${label}은(는) 마침표로만 구성하거나 마침표로 끝낼 수 없습니다.`;
        if (reservedNames.has(value.split('.', 1)[0].toUpperCase())) return `${label}에 Windows 예약 이름 "${value}"을(를) 사용할 수 없습니다.`;
        return '';
    }

    function applyRules(filename, rules) {
        let result = filename;
        for (const step of rules) {
            switch (step.action) {
                case 'split_by': {
                    const separator = step.separator || '_';
                    const parts = result.split(separator).filter(Boolean);
                    const take = String(step.take ?? 0);
                    if (take.includes(':')) {
                        const [start, end] = take.split(':').map(value => value.trim());
                        if ((start && !/^\d+$/.test(start)) || (end && !/^\d+$/.test(end))) throw new Error('범위는 1:3처럼 입력하세요.');
                        result = parts.slice(Math.max(0, start ? Number(start) - 1 : 0), Math.min(parts.length, end ? Number(end) : parts.length)).join(separator);
                    } else {
                        if (!/^-?\d+$/.test(take)) throw new Error('순번은 0, 1, -1처럼 입력하세요.');
                        const index = Number(take);
                        result = parts[index < 0 ? parts.length + index : index] || '';
                    }
                    break;
                }
                case 'strip_suffix': result = result.replace(new RegExp((step.pattern || '') + '$'), ''); break;
                case 'strip_prefix': result = result.replace(new RegExp('^' + (step.pattern || '')), ''); break;
                case 'replace': if (step.from) result = result.replaceAll(step.from, step.to || ''); break;
                case 'strip': result = result.trim(); break;
            }
        }
        return result.trim();
    }

    async function fileSha256(file) {
        const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
        return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
    }

    async function collectImageFiles(source, signal, onProgress) {
        // Capture drag entries before the browser clears the drop event's data.
        const roots = source.items
            ? Array.from(source.items).filter(item => item.kind === 'file').map(item => ({entry: item.webkitGetAsEntry?.(), file: item.getAsFile()}))
            : Array.from(source).map(file => ({file}));
        if (!roots.length && source.files) roots.push(...Array.from(source.files).map(file => ({file})));
        const files = [];
        let skipped = 0;
        function add(file, path) {
            signal.throwIfAborted();
            if (file.type.startsWith('image/') || /\.(webp|png|jpe?g|gif|bmp|avif)$/i.test(file.name)) {
                files.push({file, path: path || file.webkitRelativePath || file.name});
                if (files.length % 100 === 0) onProgress?.(files.length);
            } else skipped++;
        }
        async function read(entry) {
            signal.throwIfAborted();
            if (entry.isFile) {
                const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
                add(file, entry.fullPath.replace(/^\//, ''));
            } else if (entry.isDirectory) {
                const reader = entry.createReader();
                // Chromium returns directory entries in batches; drain every batch.
                while (true) {
                    signal.throwIfAborted();
                    const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
                    if (!batch.length) break;
                    for (const child of batch) await read(child);
                }
            }
        }
        for (const root of roots) {
            signal.throwIfAborted();
            if (root.entry) await read(root.entry);
            else if (root.file) add(root.file);
        }
        onProgress?.(files.length);
        return {files, skipped};
    }

    async function importCharacters({botName, names, entries, preflightId, request, onProgress}) {
        const errors = [];
        const result = {added: 0, uploaded: 0, replaced: 0, skipped: entries.filter(entry => entry.action === 'unchanged').length, errors};
        const uploads = entries.filter(entry => entry.action !== 'unchanged');
        const total = names.length + uploads.length;
        let completed = 0;
        let lastProgress = -Infinity;
        const progress = (message) => {
            const now = Date.now();
            // Thousands of small images must not redraw the whole import form
            // for every response. Always report stage changes and completion.
            if (!message && completed !== total && now - lastProgress < 100) return;
            lastProgress = now;
            onProgress?.(message || `처리 ${completed}/${total} · 동일 이미지 ${result.skipped}장 건너뜀`, {completed, total});
        };
        const failedNames = new Set();
        progress();
        if (names.length) {
            progress(`캐릭터 ${names.length}개 일괄 추가 중…`);
            try {
                const data = await request('bot_mode/action', {action: 'add_character', bot_name: botName, char_names: names});
                result.added = (data.added || []).length;
                for (const item of data.errors || []) {
                    failedNames.add(item.character);
                    errors.push(`${item.character}: ${item.error}`);
                }
            } catch (error) {
                names.forEach(name => failedNames.add(name));
                errors.push(error.message || String(error));
            }
            completed += names.length;
            progress();
        }

        const groups = new Map();
        for (const entry of uploads) {
            if (failedNames.has(entry.name)) { completed++; continue; }
            const key = `${entry.name}\u0000${(entry.existingFilename || entry.file.name).toLowerCase()}`;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(entry);
        }
        const queue = [...groups.values()];
        let next = 0;
        const worker = async () => {
            while (next < queue.length) {
                const group = queue[next++];
                // Replacements of the same destination must preserve selection order.
                for (const entry of group) {
                    const form = new FormData();
                    form.append('file', entry.file);
                    form.append('bot', botName);
                    form.append('character', entry.name);
                    form.append('preflight_id', preflightId);
                    form.append('preflight_index', String(entry.preflightIndex));
                    try {
                        const data = await request('bot_mode/upload', form);
                        if (data.action === 'unchanged' || data.duplicate === true) result.skipped++;
                        else if (data.action === 'replaced') result.replaced++;
                        else result.uploaded++;
                    } catch (error) { errors.push(`${entry.file.name}: ${error.message || String(error)}`); }
                    completed++;
                    progress();
                }
            }
        };
        const concurrency = Math.min(queue.length, Math.max(2, Math.min(8, Number(globalThis.navigator?.hardwareConcurrency) || 8)));
        await Promise.all(Array.from({length: concurrency}, worker));
        progress();
        return result;
    }

    async function runConcurrent(items, process, {signal, concurrency = 20} = {}) {
        // Keep enough requests ready for the server's provider gate. This is a
        // bounded producer pool, not a second copy of provider rate settings.
        let next = 0;
        const worker = async () => {
            while (!signal?.aborted && next < items.length) {
                const index = next++;
                await process(items[index], index);
            }
        };
        await Promise.all(Array.from({length: Math.min(items.length, concurrency)}, worker));
    }

    return {getStorageNameError, applyRules, fileSha256, collectImageFiles, importCharacters, runConcurrent};
})();
