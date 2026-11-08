// Representative selection and apply planning extracted from the original project.
// Both frontends use these same rules; the bot API owns persistence.
globalThis.ComfyRepBatch = (() => {
        function _repBatchProfilesFromState(profiles) {
            return (profiles || []).map((profile, index) => {
                const reps = Array.isArray(profile?.render_overrides?.rep_images)
                    ? profile.render_overrides.rep_images : [];
                return {
                    clientKey: `saved:${profile.id}`,
                    profileId: profile.id,
                    profileLabel: profile.label || `카드 ${index + 1}`,
                    rep0: reps[0] || '',
                    isNew: false,
                    useOwnSearch: false,
                    ownSearch: '',
                    manualTarget: '',
                    removePending: false,
                    _target: ''
                };
            });
        }

        function _repBatchCharacterPalette(charName) {
            let hash = 2166136261;
            for (const char of String(charName || '').normalize('NFKC')) {
                hash ^= char.codePointAt(0);
                hash = Math.imul(hash, 16777619);
            }
            const hue = (hash >>> 0) % 360;
            return {
                background: `hsla(${hue},70%,70%,0.35)`,
                header: `hsla(${hue},70%,65%,0.5)`,
                border: `hsla(${hue},60%,45%,0.85)`
            };
        }

        async function _repBatchRunLimited(tasks, limit) {
            const queue = tasks.slice();
            const n = Math.max(1, Math.min(limit || 8, queue.length || 1));
            async function worker() {
                while (queue.length) {
                    const t = queue.shift();
                    if (!t) break;
                    try { await t(); } catch (e) { /* 개별 에러는 task 내부에서 처리 */ }
                }
            }
            await Promise.all(Array.from({length: n}, () => worker()));
        }

        function _repBatchComputeDisplay(row, profile, search, limit = 10) {
            const all = row.baseOrder.slice();
            let ordered;
            if (profile.rep0) {
                const rep0Obj = all.find(im => im.filename === profile.rep0);
                const rest = all.filter(im => im.filename !== profile.rep0);
                ordered = rep0Obj ? [rep0Obj].concat(rest) : rest;
            } else {
                ordered = all;
            }
            if (search) {
                const matched = ordered.filter(im => (im.filename || '').toLowerCase().includes(search));
                if (matched.length === 0) return _repBatchCapTen(ordered, row, profile, limit); // 일치 없으면 원래 순서 유지
                const matchedSet = new Set(matched.map(im => im.filename));
                const unmatched = ordered.filter(im => !matchedSet.has(im.filename));
                let result;
                if (profile.rep0) {
                    const rep0Obj = ordered.find(im => im.filename === profile.rep0);
                    const matchedNoRep0 = matched.filter(im => im.filename !== profile.rep0);
                    const unmatchedNoRep0 = unmatched.filter(im => im.filename !== profile.rep0);
                    result = (rep0Obj ? [rep0Obj] : []).concat(matchedNoRep0).concat(unmatchedNoRep0);
                } else {
                    result = matched.concat(unmatched);
                }
                return _repBatchCapTen(result, row, profile, limit);
            }
            return _repBatchCapTen(ordered, row, profile, limit);
        }

        function _repBatchCapTen(arr, row, profile, limit = 10) {
            const forcedNames = [profile.rep0, profile.manualTarget].filter(Boolean);
            const forced = forcedNames
                .map(filename => row.baseOrder.find(im => im.filename === filename))
                .filter((image, index, values) => image && values.findIndex(item => item?.filename === image.filename) === index);
            const forcedSet = new Set(forced.map(image => image.filename));
            return forced.concat(arr.filter(image => !forcedSet.has(image.filename))).slice(0, limit);
        }

        function _repBatchTarget(row, search) {
            if (!row || !row.baseOrder || row.baseOrder.length === 0) return '';
            if (search) {
                const m = row.baseOrder.find(im => (im.filename || '').toLowerCase().includes(search));
                if (m) return m.filename;
            }
            return (row.baseOrder[0] && row.baseOrder[0].filename) || '';
        }

        function _repBatchResolvedTarget(row, profile, search) {
            const manual = String(profile?.manualTarget || '');
            if (manual && row?.baseOrder?.some(image => image.filename === manual)) return manual;
            return _repBatchTarget(row, search);
        }

        function plan(rows, commonSearch, mode) {
            const items = [];
            const skipProtected = [];
            const skipNoImage = [];
            const duplicateGroups = [];
            rows.forEach(row => {
                if (!row.loaded || row.error) {
                    (row.profiles || []).forEach(profile => skipNoImage.push(`${row.charName}/${profile.profileLabel}`));
                    return;
                }
                // 보호되어 유지되는 대표도 포함해 적용 후의 상태를 비교한다.
                const finalReps = row.profiles.map(profile =>
                    profile.removePending || profile.isNew ? '' : profile.rep0
                );
                row.profiles.forEach((profile, profileIndex) => {
                    if (profile.removePending) {
                        items.push({
                            char_name: row.charName,
                            visual_card_id: profile.profileId,
                            remove_profile: true
                        });
                        return;
                    }
                    if (!row.baseOrder || row.baseOrder.length === 0) {
                        skipNoImage.push(`${row.charName}/${profile.profileLabel}`);
                        return;
                    }
                    if (!profile.isNew && mode === 'protect' && profile.rep0 && !profile.manualTarget) {
                        skipProtected.push(`${row.charName}/${profile.profileLabel}`);
                        return;
                    }
                    const search = profile.useOwnSearch
                        ? String(profile.ownSearch || '').trim().toLowerCase() : commonSearch;
                    const target = _repBatchResolvedTarget(row, profile, search);
                    if (!target) {
                        skipNoImage.push(`${row.charName}/${profile.profileLabel}`);
                        return;
                    }
                    const item = {
                        char_name: row.charName,
                        visual_card_id: profile.profileId,
                        filename: target,
                        manual_override: !!profile.manualTarget
                    };
                    if (profile.isNew) {
                        item.create_profile = true;
                        item.profile_label = profile.profileLabel;
                    }
                    items.push(item);
                    finalReps[profileIndex] = target;
                });
                // 같은 파일명이라도 캐릭터 폴더가 다르면 별개의 이미지다.
                const cardsByImage = new Map();
                finalReps.forEach((filename, profileIndex) => {
                    if (!filename) return;
                    if (!cardsByImage.has(filename)) cardsByImage.set(filename, []);
                    const profile = row.profiles[profileIndex];
                    cardsByImage.get(filename).push(`[${profileIndex + 1}] ${profile.profileLabel}`);
                });
                cardsByImage.forEach((cards, filename) => {
                    if (cards.length > 1) {
                        duplicateGroups.push(`캐릭터: ${row.charName}\n이미지: ${filename}\n중복 카드: ${cards.join(', ')}`);
                    }
                });
            });
            return {items, skipProtected, skipNoImage, duplicateGroups};
        }

        return {profilesFromState: _repBatchProfilesFromState, characterPalette: _repBatchCharacterPalette, runLimited: _repBatchRunLimited, computeDisplay: _repBatchComputeDisplay, capTen: _repBatchCapTen, target: _repBatchTarget, resolvedTarget: _repBatchResolvedTarget, plan};
})();
