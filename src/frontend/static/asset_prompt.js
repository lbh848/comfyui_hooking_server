// One prompt builder for the original dashboard and Comfy Workshop.
globalThis.ComfyAssetPrompt = (() => {
        function normalizeAssetWorkflowType(value) {
            const normalized = String(value || '').trim().toLowerCase();
            if (normalized === 'regular') return 'ilxl';
            if (normalized === 'anima') return 'anima_ilxl';
            if (['ilxl', 'anima_ilxl', 'anima_only', 'moss'].includes(normalized)) return normalized;
            if (normalized) console.error('[ASSET] 알 수 없는 에셋 워크플로우 타입:', value);
            return 'ilxl';
        }

        function getAssetWorkflowCapabilities(type = 'ilxl') {
            const normalized = normalizeAssetWorkflowType(type);
            return {
                type: normalized,
                ilxl: normalized === 'ilxl' || normalized === 'anima_ilxl',
                anima: normalized === 'anima_ilxl' || normalized === 'anima_only',
                moss: normalized === 'moss',
                dual: normalized === 'anima_ilxl',
                pose: normalized === 'ilxl',
                ipadapter: normalized === 'ilxl',
                faceLora: normalized === 'anima_ilxl' || normalized === 'anima_only',
                seed: normalized !== 'ilxl',
            };
        }

        function filterAssetLorasForCapabilities(items, capabilities = getAssetWorkflowCapabilities()) {
            return (Array.isArray(items) ? items : []).filter(item => {
                const base = String(item?.BASE || 'anima').trim().toLowerCase();
                if (capabilities.moss) return base === 'moss';
                if (base === 'moss') return false;
                return (base === 'sdxl' || base === 'ilxl') ? capabilities.ilxl : capabilities.anima;
            });
        }

        function defaultLoraPath(l) {
            const src = l.source || 'asset';
            if (src === 'external') return l.lora_path;
            if (src === 'bot') return 'SOYA_BOT_LORA\\' + l.lora_path;
            if (src === 'instance') return 'SOYA_INSTANCE_LORA\\' + l.lora_path;
            if (src === 'style') return 'SOYA_STYLE_LORA\\' + l.lora_path;
            return 'SOYA_ASSET_LORA\\' + l.lora_path;
        }

        function getAssetPresetValues(assetTags, category) {
            if (category === 'composition_presets') return { ...assetTags.expressions, ...assetTags.composition_presets };
            return assetTags[category] || {};
        }

        function _resolveAssetPromptSlotTags(slot, assetTags) {
            const capabilities = getAssetWorkflowCapabilities(slot.asset_workflow_type);
            // 활성화된 LoRA 중 현재 에셋 프로필과 호환되는 BASE만 병합한다.
            const triggerSrc = filterAssetLorasForCapabilities([
                ...(slot.lora_enabled ? (slot.lora_list || []) : []),
                ...(slot.style_lora_enabled ? (slot.style_lora_list || []) : []),
            ], capabilities);
            const loraTriggerWords = triggerSrc.map(l => l.trigger).filter(t => t && t.trim()).join(', ');
            const animaLoraTriggerWords = triggerSrc.filter(l => (l.BASE || 'anima') === 'anima').map(l => l.trigger).filter(t => t && t.trim()).join(', ');
            const mossLoraTriggerWords = triggerSrc.filter(l => String(l.BASE || '').trim().toLowerCase() === 'moss').map(l => l.trigger).filter(t => t && t.trim()).join(', ');
            const sdxlLoraTriggerWords = triggerSrc.filter(l => l.BASE === 'sdxl').map(l => l.trigger).filter(t => t && t.trim()).join(', ');
            const artistTags = Array.isArray(slot.raw_artist_tags)
                ? slot.raw_artist_tags
                : ((slot.artist_preset && assetTags.artist_presets?.[slot.artist_preset]) || []);
            const animaArtistTags = Array.isArray(slot.raw_anima_artist_tags)
                ? slot.raw_anima_artist_tags
                : ((slot.anima_artist_preset && assetTags.artist_presets?.[slot.anima_artist_preset]) || []);
            const qTags = Array.isArray(slot.raw_quality_tags)
                ? slot.raw_quality_tags
                : ((slot.quality_preset && assetTags.quality_presets?.[slot.quality_preset]) || []);
            const cTags = Array.isArray(slot.raw_composition_tags)
                ? slot.raw_composition_tags
                : ((slot.composition_preset && getAssetPresetValues(assetTags, 'composition_presets')[slot.composition_preset]) || []);
            const appTags = Array.isArray(slot.raw_appearance_tags)
                ? slot.raw_appearance_tags
                : ((slot.appearance && assetTags.appearances?.[slot.appearance]) || []);
            const exprTags = Array.isArray(slot.raw_expression_tags)
                ? slot.raw_expression_tags
                : ((slot.expression && assetTags.expressions?.[slot.expression]) || []);
            const outfitTags = Array.isArray(slot.raw_outfit_tags)
                ? slot.raw_outfit_tags
                : ((slot.outfit && assetTags.outfits?.[slot.outfit]) || []);
            const backgroundTags = Array.isArray(slot.raw_background_tags)
                ? slot.raw_background_tags
                : ((slot.background_preset && assetTags.background_presets?.[slot.background_preset]) || []);
            const naturalLanguage = slot.natural_language_preset
                ? (assetTags?.natural_language_presets?.[slot.natural_language_preset] || slot.natural_language || '')
                : (slot.natural_language || '');
            const cnTags = Array.isArray(slot.raw_character_negative_tags)
                ? slot.raw_character_negative_tags
                : ((slot.character_negative_preset && assetTags.character_negative_presets?.[slot.character_negative_preset]) || []);
            const nTags = Array.isArray(slot.raw_negative_tags)
                ? slot.raw_negative_tags
                : ((slot.negative_preset && assetTags.negative_presets?.[slot.negative_preset]) || []);
            const animaQTags = Array.isArray(slot.raw_anima_quality_tags)
                ? slot.raw_anima_quality_tags
                : ((slot.anima_quality_preset && assetTags.quality_presets?.[slot.anima_quality_preset]) || []);
            const animaNTags = Array.isArray(slot.raw_anima_negative_tags)
                ? slot.raw_anima_negative_tags
                : ((slot.anima_negative_preset && assetTags.negative_presets?.[slot.anima_negative_preset]) || []);
            return { loraTriggerWords, animaLoraTriggerWords, mossLoraTriggerWords, sdxlLoraTriggerWords, artistTags, animaArtistTags, qTags, cTags, appTags, exprTags, outfitTags, backgroundTags, naturalLanguage, cnTags, nTags, animaQTags, animaNTags };
        }

        async function buildAssetPromptData(slot, assetTags, loadPose, resolveLoraPath = defaultLoraPath) {
            if (!assetTags) return null;
            const t = _resolveAssetPromptSlotTags(slot, assetTags);
            const capabilities = getAssetWorkflowCapabilities(slot.asset_workflow_type);
            const isAnima = capabilities.anima;
            const isMoss = capabilities.moss;
            const isStructured = isAnima || isMoss;
            const isDual = capabilities.dual;
            const poseEnabled = capabilities.pose && (slot.pose_enabled || false);
            const primaryLoraTriggerWords = isMoss ? t.mossLoraTriggerWords : t.animaLoraTriggerWords;

            let positiveTags = '';
            if (isStructured) {
                const animaQuality = t.animaQTags.filter(v => v.trim());
                const animaArtist = t.animaArtistTags.filter(v => v.trim());
                const sdxlQuality = t.qTags.filter(v => v.trim());
                const sdxlArtist = t.artistTags.filter(v => v.trim());
                const section1Rest = [
                    ...(primaryLoraTriggerWords ? [primaryLoraTriggerWords] : []),
                    ...t.animaArtistTags, ...t.animaQTags,
                    ...t.cTags, ...t.appTags, ...t.exprTags, ...t.outfitTags, ...t.backgroundTags,
                    ...(t.naturalLanguage ? [t.naturalLanguage] : []),
                ].filter(v => typeof v === 'string' && v.trim());
                const section2Rest = [
                    ...(t.sdxlLoraTriggerWords ? [t.sdxlLoraTriggerWords] : []),
                    ...t.artistTags, ...t.qTags,
                    ...t.cTags, ...t.appTags, ...t.exprTags, ...t.outfitTags, ...t.backgroundTags,
                ].filter(v => typeof v === 'string' && v.trim());
                positiveTags = '[ANIMA_QUALITY]\n' + animaQuality.join(', ')
                    + '\n[ANIMA_ARTIST]\n' + animaArtist.join(', ')
                    + '\n[ANIMA]\n' + section1Rest.join(', ')
                    + (isDual ? '\n[SDXL_QUALITY]\n' + sdxlQuality.join(', ')
                        + '\n[SDXL_ARTIST]\n' + sdxlArtist.join(', ')
                        + '\n[SDXL]\n' + section2Rest.join(', ') : '')
                    + '\n[CHAR_LIST]\nasset_mode';
            } else {
                positiveTags = [
                    ...(t.loraTriggerWords ? [t.loraTriggerWords] : []),
                    ...t.artistTags, ...t.qTags, ...t.cTags, ...t.appTags, ...t.exprTags, ...t.outfitTags, ...t.backgroundTags,
                    ...(t.naturalLanguage ? [t.naturalLanguage] : []),
                ].filter(v => typeof v === 'string' && v.trim()).join(', ');
            }

            let positiveFull = positiveTags;
            if (capabilities.anima) {
                positiveFull += '\n[SNAPREF]\n' + JSON.stringify({
                    enabled: !!slot.snapref_enabled, strength: slot.snapref_strength ?? 0,
                    mode: slot.snapref_mode || 'char', megapixels: slot.snapref_megapixels ?? 1,
                    strength_scale: slot.snapref_strength_scale || 'log', fidelity: slot.snapref_fidelity ?? 1,
                    preprocess: slot.snapref_preprocess || 'none', secondary_reference: slot.snapref_secondary_reference || 'none',
                    dir: slot.snapref_enabled ? (slot.snapref_dir || '') : '',
                });
            }
            positiveFull += `\n[FACE_ID_ACTIVATE]\n${capabilities.ipadapter && slot.ref_enabled ? 'true' : 'false'}`;
            positiveFull += `\n[FACE_ID_STR]\n${slot.ref_strength || 0.55}`;
            positiveFull += `\n[FACE_ID_DIR]\n${slot.ref_dir || 'soya_char_ref/fallback'}`;
            positiveFull += `\n[STYLE_ACTIVATE]\n${capabilities.ipadapter && slot.style_ref_enabled ? 'true' : 'false'}`;
            positiveFull += `\n[STYLE_STR]\n${slot.style_ref_strength || 0.55}`;
            positiveFull += `\n[STYLE_DIR]\n${slot.style_ref_dir || 'soya_style_ref/fallback'}`;
            // FACE_CROP (ANIMA only)
            if (isAnima) {
                positiveFull += `\n[FACE_CROP_TOP]\n${slot.face_crop_top ?? 2.5}`;
                positiveFull += `\n[FACE_CROP_BOTTOM]\n${slot.face_crop_bottom ?? 1}`;
            }
            const compatibleRegularLoras = filterAssetLorasForCapabilities(slot.lora_list, capabilities);
            const compatibleStyleLoras = filterAssetLorasForCapabilities(slot.style_lora_list, capabilities);
            const loraActive = slot.lora_enabled && compatibleRegularLoras.length > 0;
            const styleLoraActive = slot.style_lora_enabled && compatibleStyleLoras.length > 0;
            const regularLoraData = compatibleRegularLoras.map(l => ({lora_path: resolveLoraPath(l), str: l.strength, BASE: l.BASE || (isMoss ? 'moss' : 'anima')}));
            const styleLoraData = compatibleStyleLoras.map(l => ({lora_path: resolveLoraPath(l), str: l.strength, BASE: l.BASE || (isMoss ? 'moss' : 'anima')}));
            if (isStructured) {
                positiveFull += `\n[LORA_ACTIVATE]\n${loraActive ? 'true' : 'false'}`;
                positiveFull += `\n[LORA_DATA]\n${JSON.stringify({list: regularLoraData})}`;
            } else {
                // ILXL: 스타일 LoRA를 LORA_DATA에 합쳐서 단일 로더 처리 (둘 중 하나만 ON이어도 활성화)
                const mergedActive = loraActive || styleLoraActive;
                positiveFull += `\n[LORA_ACTIVATE]\n${mergedActive ? 'true' : 'false'}`;
                positiveFull += `\n[LORA_DATA]\n${JSON.stringify({list: [...regularLoraData, ...styleLoraData]})}`;
            }
            // Face LoRA (ANIMA only)
            if (isAnima) {
                const compatibleFaceLoras = filterAssetLorasForCapabilities(slot.face_lora_list, capabilities);
                const faceLoraActive = slot.face_lora_enabled && compatibleFaceLoras.length > 0;
                const slotFaceLoraUpscale = (slot.face_lora_upscale_size || '').toString().trim();
                positiveFull += `\n[FACE_LORA_ACTIVATE]\n${faceLoraActive ? 'true' : 'false'}`;
                if (faceLoraActive) {
                    positiveFull += `\n[FACE_LORA_DATA]\n${JSON.stringify({list: compatibleFaceLoras.map(l => {
                        const item = {lora_path: resolveLoraPath(l), str: l.strength, BASE: l.BASE || 'anima', CHAR: 'asset_mode'};
                        if (slotFaceLoraUpscale) item.UPSCALE_SIZE = slotFaceLoraUpscale;
                        return item;
                    })})}`;
                } else {
                    positiveFull += `\n[FACE_LORA_DATA]\n{"list":[]}`;
                }
                // CHAR_FACE_TAG_INFORM
                const faceLoraList = compatibleFaceLoras;
                const batchFaceAnimaTriggers = faceLoraList.filter(l => (l.BASE || 'anima') === 'anima').map(l => l.trigger).filter(t => t && t.trim()).join(', ');
                const batchFaceSdxlTriggers = faceLoraList.filter(l => l.BASE === 'sdxl').map(l => l.trigger).filter(t => t && t.trim()).join(', ');
                positiveFull += `\n[CHAR_FACE_TAG_INFORM]\n${JSON.stringify({list: [{FACE_TAGS: slot.face_tags || '', EYE_TAGS: slot.eye_tags || '', POSITIVE: "", TRIGGER_ANIMA: batchFaceAnimaTriggers, TRIGGER_SDXL: batchFaceSdxlTriggers, CHAR: "asset_mode"}]})}`;
            }
            // Style(그림체) LoRA — ANIMA·MOSS는 별도 토큰(분리 로더).
            // ILXL 모드는 스타일 LoRA가 LORA_DATA에 흡수되므로 이 토큰들을 출력하지 않음.
            if (isStructured) {
                positiveFull += `\n[STYLE_LORA_ACTIVATE]\n${styleLoraActive ? 'true' : 'false'}`;
                positiveFull += `\n[STYLE_LORA_DATA]\n${JSON.stringify({list: styleLoraData})}`;
            }
            positiveFull += `\n[POSE_ACTIVATE]\n${poseEnabled ? 'true' : 'false'}`;
            if (poseEnabled && slot.pose_id) {
                try {
                    const poseResult = await loadPose(slot.pose_id);
                    const poseJson = poseResult.keypoints || poseResult;
                    positiveFull += `\n[POSE_DATA]\n${JSON.stringify(poseJson)}`;
                } catch (e) {
                    throw new Error(`Pose ${slot.pose_id}: ${e.message}`);
                }
            } else {
                positiveFull += `\n[POSE_DATA]\n{"people":[{"pose_keypoints_2d":[381.10,379.86,1,413.21,506.58,1,223.64,436.41,1,55.33,624.12,1,64.77,454.24,1,579.09,575.30,1,362.22,935.42,1,331.06,972.15,1,196.97,797.67,1,264.96,1013.47,1,257.41,1024.49,1,350.89,820.63,1,419.82,1011.63,1,419.82,1024.49,1,315.95,343.13,1,462.31,294.46,1,263.07,365.17,1,546.35,301.80,1]}],"canvas_height":1024,"canvas_width":700}`;
            }
            positiveFull += `\n[HRF_ACTIVATE]\n${capabilities.ilxl && slot.hrf_sdxl ? 'true' : 'false'}`;
            if (isAnima) positiveFull += `\n[ANIMA_HRF_ACTIVATE]\n${slot.hrf_anima ? 'true' : 'false'}`;
            if (isMoss) positiveFull += `\n[MOSS_HRF_ACTIVATE]\n${slot.hrf_moss ? 'true' : 'false'}`;
            positiveFull += `\n[HRF_SIZE]\n${slot.hrf_size || 2.0}`;
            positiveFull += `\n[HRF_RESTORE_SIZE]\n${slot.hrf_restore_size !== false ? 'true' : 'false'}`;
            positiveFull += `\n[HRF_CONTROL_NET]\n${slot.hrf_control_net ? 'true' : 'false'}`;
            positiveFull += `\n[IMG_W]\n${slot.img_w || 700}`;
            positiveFull += `\n[IMG_H]\n${slot.img_h || 1024}`;
            if (isAnima) {
                positiveFull += `\n[ANIMA_FD_ACTIVATE]\n${slot.anima_fd_enabled ? 'true' : 'false'}`;
                positiveFull += `\n[ANIMA_HD_ACTIVATE]\n${slot.anima_hd_enabled ? 'true' : 'false'}`;
                positiveFull += `\n[ANIMA_ED_ACTIVATE]\n${slot.anima_ed_enabled ? 'true' : 'false'}`;
            }
            positiveFull += `\n[FD_ACTIVATE]\n${capabilities.ilxl && slot.sdxl_fd_enabled ? 'true' : 'false'}`;
            positiveFull += `\n[HD_ACTIVATE]\n${capabilities.ilxl && slot.sdxl_hd_enabled ? 'true' : 'false'}`;
            positiveFull += `\n[ED_ACTIVATE]\n${capabilities.ilxl && slot.sdxl_ed_enabled ? 'true' : 'false'}`;
            if (capabilities.seed) {
                positiveFull += `\n[SEED]\n${slot.seed ?? -1}`;
            }
            positiveFull += '\n[END]';

            let negative;
            if (isStructured) {
                const animaNegParts = [...t.cnTags, ...t.animaNTags].filter(v => v.trim());
                const sdxlNegParts = [...t.cnTags, ...t.nTags].filter(v => v.trim());
                // ANIMA-only 워크플로도 이 구분자를 ANIMA 부정 프롬프트의 끝 표식으로 사용한다.
                negative = animaNegParts.join(', ') + '\n[SDXL]\n'
                    + (isDual ? sdxlNegParts.join(', ') : '');
            } else {
                negative = [...t.cnTags, ...t.nTags].filter(v => v.trim()).join(', ');
            }

            let document;
            if (isStructured) {
                const fields = {};
                const removed = new Set(['END', 'POSE_ACTIVATE', 'POSE_DATA', 'HRF_CONTROL_NET',
                    'FACE_ID_ACTIVATE', 'FACE_ID_STR', 'FACE_ID_DIR', 'STYLE_ACTIVATE', 'STYLE_STR', 'STYLE_DIR']);
                const blocks = [...positiveFull.matchAll(/^\[([A-Z][A-Z0-9_]*)\][ \t]*(?:\r?\n|$)/gm)];
                for (const [index, match] of blocks.entries()) {
                    const key = match[1];
                    if (removed.has(key)) continue;
                    const value = positiveFull.slice(match.index + match[0].length, blocks[index + 1]?.index ?? positiveFull.length).trim();
                    fields[key] = key === 'CHAR_LIST' ? value.split(',').map(name => name.trim()).filter(Boolean)
                        : key.endsWith('_ACTIVATE') || key === 'HRF_RESTORE_SIZE' ? value === 'true'
                        : ['IMG_W', 'IMG_H', 'SEED', 'FACE_CROP_TOP', 'FACE_CROP_BOTTOM', 'HRF_SIZE'].includes(key) ? Number(value)
                        : key.endsWith('_DATA') || ['CHAR_FACE_TAG_INFORM', 'SNAPREF'].includes(key) ? JSON.parse(value)
                        : value;
                }
                fields.ANIMA_CONTENT = [primaryLoraTriggerWords, ...t.animaArtistTags, ...t.cTags, ...t.appTags, ...t.exprTags, ...t.outfitTags, ...t.backgroundTags, t.naturalLanguage].filter(value => typeof value === 'string' && value.trim()).join(', ');
                delete fields.ANIMA;
                if ('SDXL' in fields) { fields.SDXL_CONTENT = fields.SDXL; delete fields.SDXL; }
                const [animaNegative, sdxlNegative = ''] = negative.split('[SDXL]');
                fields.NEGATIVE_ANIMA = animaNegative.trim();
                fields.NEGATIVE_SDXL = sdxlNegative.trim();
                fields.MULTI_CHAR = { enable: false };
                document = { schema: 'soya.illustration.v2', fields, assets: {} };
            }
            return { positive: positiveFull, negative, positiveLength: positiveTags.length, document };
        }

        function assetMaskPlacements(regions, slots, assetTags) {
            return regions.map((region, index) => {
                const dx = region.x + region.width / 2 - .5;
                const dy = region.y + region.height / 2 - .5;
                const position = Math.max(Math.abs(dx), Math.abs(dy)) <= 1 / 6 ? 'middle'
                    : Math.abs(dx) >= Math.abs(dy) ? (dx < 0 ? 'left' : 'right') : (dy < 0 ? 'top' : 'bottom');
                const appearance = _resolveAssetPromptSlotTags(slots[index], assetTags).appTags.join(', ');
                const subjects = [...new Set(Array.from(appearance.matchAll(/(?:^|[,\s(])(1(?:girl|boy|other))(?=$|[,\s):])/gi), match => match[1].toLowerCase()))];
                const subject = subjects.length === 1 ? subjects[0] : 'person';
                return `The ${subject} on the ${position}`;
            });
        }

        async function buildAssetMaskPromptData(slots, assetTags, layout) {
            if (!Array.isArray(slots) || slots.length < 1 || slots.length > 3 || layout.regions?.length !== slots.length) throw new Error('영역과 캐릭터를 1~3개 지정하세요.');
            const composition = layout.composition_prompt?.trim() || '';
            const documents = await Promise.all(slots.map(slot => buildAssetPromptData(slot, assetTags, async () => ({}))));
            if (documents.some(value => !value?.document)) throw new Error('마스크 기반 생성은 ANIMA, ANIMA+ILXL 또는 MOSS 모델을 사용하세요.');
            const document = structuredClone(documents[0].document);
            const fields = document.fields;
            const parts = slots.map(slot => _resolveAssetPromptSlotTags(slot, assetTags));
            const styleMask = slots.length === 1 && slots[0].mask_area_limit_enabled;
            const clean = values => values.filter(value => typeof value === 'string' && value.trim()).map(value => value.trim());
            const unique = values => [...new Set(clean(values))];
            const background = clean(parts[0].backgroundTags).join(', ');
            const characterText = part => clean([...part.cTags, ...part.appTags, ...part.exprTags, ...part.outfitTags]).join(', ');
            const inform = parts.map(characterText);
            const regionalText = parts.map(part => part.naturalLanguage.trim());
            if (styleMask && !inform[0] && !regionalText[0]) inform[0] = 'character';
            if (inform.some((value, index) => !value && !regionalText[index])) throw new Error('각 영역의 외모·복장·표정 또는 자연어를 입력하세요.');
            const names = layout.regions.map(region => region.name);
            const primaryCapabilities = getAssetWorkflowCapabilities(slots[0].asset_workflow_type);
            const primaryBase = primaryCapabilities.moss ? 'moss' : 'anima';
            const styleTriggers = unique(filterAssetLorasForCapabilities(slots[0].style_lora_enabled ? slots[0].style_lora_list : [], primaryCapabilities)
                .filter(item => String(item.BASE || 'anima').trim().toLowerCase() === primaryBase).map(item => item.trigger));
            const triggers = slots.map(slot => {
                const slotCapabilities = getAssetWorkflowCapabilities(slot.asset_workflow_type);
                const slotBase = slotCapabilities.moss ? 'moss' : 'anima';
                return unique([...filterAssetLorasForCapabilities(slot.lora_enabled ? slot.lora_list : [], slotCapabilities)
                    .filter(item => String(item.BASE || 'anima').trim().toLowerCase() === slotBase).map(item => item.trigger), ...styleTriggers]);
            });
            const shared = unique([...parts[0].animaArtistTags, ...parts[0].animaQTags, background]);
            fields.CHAR_LIST = names;
            fields.MULTI_CHAR = { enable: true, char_num: names.length, char_name_list: names, char_inform: inform,
                character_labels: Object.fromEntries(names.map((name, index) => [name, slots[index].character])),
                char_trigger_list: triggers, background_trigger_list: styleTriggers,
                ...(styleMask ? { mask_area_limit_enabled: true } : {}),
                background_prompt: styleMask && !shared.length && !styleTriggers.length ? 'background' : background,
                composition_prompt: composition, char_composition_list: regionalText,
                shared_tag: { before_char: shared, after_char: [] }, layout: structuredClone(layout) };
            // Character appearance/wardrobe stays local. Only style and background are shared.
            if (!styleMask) {
                fields.ANIMA_CONTENT = clean([...styleTriggers, ...parts[0].animaArtistTags, background,
                    composition, ...inform.map((value, index) => clean([...triggers[index], value]).join(', '))]).join(', ');
                fields.NEGATIVE_ANIMA = unique([...parts[0].animaNTags, ...parts.flatMap(part => part.cnTags)]).join(', ');
                fields.NEGATIVE_SDXL = unique([...parts[0].nTags, ...parts.flatMap(part => part.cnTags)]).join(', ');
                if (primaryCapabilities.dual) {
                    fields.SDXL_CONTENT = unique([...parts[0].artistTags, ...parts[0].qTags, background,
                        composition, ...parts.map((part, index) => clean([part.sdxlLoraTriggerWords, inform[index]]).join(', '))]).join(', ');
                }
            }
            for (const [dataKey, activeKey] of [['LORA_DATA', 'LORA_ACTIVATE'], ['FACE_LORA_DATA', 'FACE_LORA_ACTIVATE'], ['CHAR_FACE_TAG_INFORM', null]]) {
                fields[dataKey] = { list: documents.flatMap((value, index) =>
                    (!activeKey || value.document.fields[activeKey] ? value.document.fields[dataKey]?.list || [] : [])
                        .map(item => ({ ...item, CHAR: names[index] }))) };
                if (activeKey) fields[activeKey] = fields[dataKey].list.length > 0;
            }
            fields.SNAPREF = { ...fields.SNAPREF, enabled: primaryCapabilities.anima && slots.some(slot => slot.snapref_enabled), references: [] };
            delete fields.SNAPREF.dir;
            return document;
        }

        function migrateBatchChainFields(c) {
            if (c.hrf_enabled !== undefined && c.hrf_sdxl === undefined) {
                c.hrf_sdxl = !!c.hrf_enabled;
                c.hrf_anima = !!c.hrf_enabled;
                delete c.hrf_enabled;
                if (c.hrf_mode === 'anima') { c.hrf_sdxl = false; c.hrf_anima = true; }
                else if (c.hrf_mode === 'sdxl') { c.hrf_sdxl = true; c.hrf_anima = false; }
                delete c.hrf_mode;
            }
            if (c.fd_enabled !== undefined && c.sdxl_fd_enabled === undefined) {
                c.sdxl_fd_enabled = !!c.fd_enabled;
                c.sdxl_hd_enabled = !!c.hd_enabled;
                c.sdxl_ed_enabled = !!c.ed_enabled;
                c.anima_fd_enabled = false;
                c.anima_hd_enabled = false;
                c.anima_ed_enabled = false;
                delete c.fd_enabled; delete c.hd_enabled; delete c.ed_enabled;
            }
        }

    return { migrateBatchChainFields, normalizeAssetWorkflowType, getAssetWorkflowCapabilities, getAssetPresetValues, filterAssetLorasForCapabilities, buildAssetPromptData, assetMaskPlacements, buildAssetMaskPromptData };
})();
