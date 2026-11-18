// Shared rules from the original profile metadata editor. Persistence stays in the bot API.
globalThis.ComfyVisualGuide = (() => {
    function identity(character, profileId) {
        return `${String(character || '').toLocaleLowerCase()}\u0000${String(profileId || '')}`;
    }
    function missing(target, analyzeImages = true) {
        return !target.currentAliases.length || !target.currentGuide || (analyzeImages && !target.currentVisualContext);
    }
    function target(botName, character, profile, profileIndex, focusIdentity = '') {
        const repImages = Array.isArray(profile.render_overrides?.rep_images)
            ? profile.render_overrides.rep_images.map(value => String(value || '').trim()).filter(Boolean) : [];
        const repFilename = repImages[0] || '';
        const result = {
            identity: identity(character, profile.id), character,
            profileId: String(profile.id || ''), profileLabel: String(profile.label || profile.id || `카드 ${profileIndex + 1}`),
            profileIndex, repFilename,
            imageUrl: repFilename ? `/api/bot_mode/image/${encodeURIComponent(botName)}/${encodeURIComponent(character)}/${encodeURIComponent(repFilename)}` : '',
            currentAliases: Array.isArray(profile.aliases) ? [...profile.aliases] : [],
            currentGuide: String(profile.selection_guide || '').trim(), currentVisualContext: String(profile.visual_context || '').trim(),
            selected: false, focused: identity(character, profile.id) === focusIdentity, suggestion: null,
        };
        result.selected = missing(result) || result.focused;
        return result;
    }
    function matches(target, query) {
        return [target.character, target.profileLabel, target.profileId, target.repFilename, ...target.currentAliases]
            .join(' ').toLocaleLowerCase().includes(String(query || '').trim().toLocaleLowerCase());
    }
    function merge(targets, suggestions) {
        const byIdentity = new Map(targets.map(target => [target.identity, target]));
        let count = 0;
        for (const suggestion of suggestions) {
            const target = byIdentity.get(identity(suggestion?.character, suggestion?.profile_id));
            if (!target) continue;
            const previous = target.suggestion;
            const next = {
                aliases: Array.isArray(suggestion.aliases) ? [...suggestion.aliases] : previous?.aliases || [],
                ...Object.fromEntries(['selection_guide', 'selection_guide_english', 'evidence', 'visual_context', 'visual_context_english',
                    'visual_context_status', 'visual_context_error', 'visual_context_image'].map(key => [key,
                    key in suggestion ? String(suggestion[key] || '') : previous?.[key] || ''])),
                confidence: ['high', 'medium', 'low'].includes(suggestion.confidence) ? suggestion.confidence : previous?.confidence || 'low',
            };
            if (JSON.stringify(previous) === JSON.stringify(next)) continue;
            target.suggestion = next;
            count++;
        }
        return count;
    }
    function metadataItems(targets) {
        return targets.map(target => ({
            character: target.character, profile_id: target.profileId, aliases: target.suggestion.aliases || [],
            selection_guide: String(target.suggestion.selection_guide || '').trim(),
            selection_guide_english: String(target.suggestion.selection_guide_english || '').trim(),
            ...(target.suggestion.visual_context_status === 'ok' ? {
                visual_context: String(target.suggestion.visual_context || '').trim(),
                visual_context_english: String(target.suggestion.visual_context_english || '').trim(),
            } : {}),
        }));
    }
    return { identity, missing, target, matches, merge, metadataItems };
})();
