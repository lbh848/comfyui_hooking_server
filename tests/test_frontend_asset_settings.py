from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_asset_select_helpers_support_native_character_select():
    source = _frontend_source()
    get_select = _function_source(
        source, "getAssetSelectValue(selectId)", "setAssetSelectValue(selectId, value)"
    )
    set_select = _function_source(
        source, "setAssetSelectValue(selectId, value)", "bindAssetCustomSelects()"
    )

    assert "document.getElementById(selectId)" in get_select
    assert "nativeSelect.value" in get_select
    assert "document.getElementById(selectId)" in set_select
    assert "nativeSelect.value = normalizedValue;" in set_select
    assert "return true;" in set_select


def test_asset_settings_restore_waits_for_data_without_fixed_timer():
    source = _frontend_source()
    load_settings = _function_source(
        source, "loadAssetSettings()", "previewAssetPrompt()"
    )

    assert "async function loadAssetSettings()" in source
    assert "await loadAssetData();" in load_settings
    assert "setTimeout(" not in load_settings
    assert "restoreSelect('asset-character-select'" in load_settings
    assert "restoreSelect('asset-appearance-select'" in load_settings
    assert "restoreSelect('asset-outfit-select'" in load_settings
    assert "restoreSelect('asset-expression-select'" in load_settings
    assert load_settings.index("restoreSelect('asset-character-select'") < load_settings.index(
        "restoreSelect('asset-appearance-select'"
    )
    assert "console.error('[ASSET] 설정 불러오기 실패: 캐시 접근 오류'" in load_settings
    assert "console.error('[ASSET] 설정 불러오기 실패:'" in load_settings


def test_asset_settings_cache_marks_fixed_character_format():
    source = _frontend_source()
    save_settings = _function_source(
        source, "saveAssetSettings()", "loadAssetSettings()"
    )

    assert "cache_version: 2" in save_settings
    assert "character: getAssetSelectValue('asset-character-select')" in save_settings
    assert "console.error('[ASSET] 설정 캐시 저장 실패:'" in save_settings


def test_asset_settings_wait_for_lightweight_preset_requests():
    source = _frontend_source()
    load_settings = _function_source(
        source, "loadAssetSettings()", "previewAssetPrompt()"
    )

    for loader in (
        "loadQualityPreset",
        "loadCompositionPreset",
        "loadCharacterNegativePreset",
        "loadNegativePreset",
        "loadAnimaQualityPreset",
        "loadAnimaNegativePreset",
    ):
        assert f"await {loader}()" in load_settings

    assert source.count("return modifyTagLightweight('load_") >= 6
    modify_tags = _function_source(
        source,
        "modifyTagLightweight(action, params, containerId, removeFnName, tagKey, syncFn)",
        "loadQualityPreset()",
    )
    assert "return true;" in modify_tags
    assert "return false;" in modify_tags
    assert "console.error(`[ASSET] 태그 프리셋 적용 예외:" in modify_tags


def test_batch_bulk_edit_loads_saved_asset_settings_into_form_before_apply():
    source = _frontend_source()
    load_bulk_settings = _function_source(
        source,
        "loadAssetSettingsIntoBatchBulkEdit()",
        "openBulkRefGalleryPicker(mode)",
    )

    assert (
        'class="batch-bulk-load-btn" '
        'onclick="loadAssetSettingsIntoBatchBulkEdit()"'
    ) in source
    assert "localStorage.getItem('assetModeSettings')" in load_bulk_settings
    assert "bulkDropdownValues[field] = normalizedValue;" in load_bulk_settings
    assert (
        "setBulkAction('bulk-ref-action', s.ref_enabled"
        in load_bulk_settings
    )
    assert (
        "setBulkAction('bulk-style-lora-action', s.style_lora_enabled"
        in load_bulk_settings
    )
    assert (
        "setBulkNumber('bulk-img-w', s.img_w, '이미지 너비', "
        "'bulk-img-size-check')"
        in load_bulk_settings
    )
    assert "bulkLoraList = cachedLoras;" in load_bulk_settings
    assert "bulkStyleLoraList = cachedStyleLoras;" in load_bulk_settings
    assert "bulkFaceLoraList = cachedFaceLoras;" in load_bulk_settings
    assert "console.error('[BATCH] 저장 설정 불러오기 실패:" in load_bulk_settings
    assert "console.error('[BATCH] 저장 설정 불러오기 예외:'" in load_bulk_settings

    # 불러오기 단계는 모달 값만 변경하고, 실제 슬롯 적용은 별도 버튼에 남겨둔다.
    assert "batchChains.forEach" not in load_bulk_settings
    assert "renderBatchChain()" not in load_bulk_settings


def test_batch_bulk_saved_disabled_face_lora_remains_disabled_on_apply():
    source = _frontend_source()
    apply_bulk_settings = _function_source(
        source,
        "applyBatchBulkEdit()",
        "duplicateBatchSlot(idx)",
    )

    list_apply = (
        "if (bulkFaceLoraList.length > 0) {\n"
        "                    slot.face_lora_list = bulkFaceLoraList.map(l => ({...l}));\n"
        "                    slot.face_lora_enabled = true;\n"
        "                }"
    )
    assert list_apply in apply_bulk_settings
    assert apply_bulk_settings.index(list_apply) < apply_bulk_settings.index(
        "if (faceLoraAction === 'off') slot.face_lora_enabled = false;",
        apply_bulk_settings.index(list_apply),
    )


def test_asset_generation_options_are_shared_across_workflows():
    source = _frontend_source()
    build_prompt = _function_source(
        source, "buildAssetPromptFromUI()", "saveAssetSettings()"
    )

    assert 'id="asset-generation-options-wrapper"' in source
    assert "<strong>생성 옵션</strong>" in source
    assert 'id="asset-sdxl-options-wrapper"' not in source
    assert 'id="asset-anima-options-wrapper"' not in source
    assert "availability === 'ilxl-only'" not in source
    for removed_id in (
        "asset-sdxl-hrf-toggle",
        "asset-sdxl-hrf-size",
        "asset-sdxl-hrf-restore-size",
        "asset-sdxl-hrf-control-net",
        "asset-sdxl-fd-toggle",
        "asset-sdxl-hd-toggle",
        "asset-sdxl-ed-toggle",
    ):
        assert removed_id not in source
    for shared_id in (
        "asset-hrf-sdxl",
        "asset-hrf-anima",
        "asset-hrf-size",
        "asset-sdxl-fd",
        "asset-sdxl-hd",
        "asset-sdxl-ed",
        "asset-anima-fd",
        "asset-anima-hd",
        "asset-anima-ed",
    ):
        assert f"getElementById('{shared_id}')" in build_prompt


def test_lv2_asset_navigation_resolves_storage_names_without_stale_selection():
    source = _frontend_source()
    navigate = _function_source(
        source,
        "navigateToImages(charName, outfit, expression)",
        "syncAssetSelectFromDirname(selectId, dirname)",
    )
    sync_select = _function_source(
        source,
        "syncAssetSelectFromDirname(selectId, dirname)",
        "navigateAssetBreadcrumb(level)",
    )

    assert "syncAssetSelectFromDirname('asset-outfit-select', outfit);" in navigate
    assert "syncAssetSelectFromDirname('asset-expression-select', expression);" in navigate
    assert "if (!getAssetSelectValue('asset-outfit-select'))" not in navigate
    assert "if (!getAssetSelectValue('asset-expression-select'))" not in navigate
    assert "onAssetOutfitChange(true);" in navigate
    assert "onAssetExpressionChange(true);" in navigate

    assert "opt.dataset.value === normalizedDirname" in sync_select
    assert "value.replace(re, '').trim() === normalizedDirname" in sync_select
    assert "matches.length === 1" in sync_select
    assert "matches.length > 1" in sync_select
    assert "setAssetSelectValue(selectId, '');" in sync_select
    assert "Lv2 선택 동기화 실패: 폴더명 비어 있음" in sync_select
    assert "Lv2 선택 동기화 실패: 일치 옵션 없음" in sync_select
    assert "return true;" in sync_select


def test_lv2_asset_navigation_reloads_images_once_after_atomic_selection_sync():
    source = _frontend_source()
    navigate = _function_source(
        source,
        "navigateToImages(charName, outfit, expression)",
        "syncAssetSelectFromDirname(selectId, dirname)",
    )
    outfit_change = _function_source(
        source,
        "onAssetOutfitChange(skipImageReload = false)",
        "addAssetOutfit()",
    )
    expression_change = _function_source(
        source,
        "onAssetExpressionChange(skipImageReload = false)",
        "addAssetExpression()",
    )

    assert navigate.count("loadAssetImages();") == 1
    assert "assetNavLevel === 2 && !skipImageReload" in outfit_change
    assert "assetNavLevel === 2 && !skipImageReload" in expression_change


def test_lv2_navigation_keeps_storage_names_separate_from_automatch_chain_names():
    source = _frontend_source()
    navigate = _function_source(
        source,
        "navigateToImages(charName, outfit, expression)",
        "syncAssetSelectFromDirname(selectId, dirname)",
    )
    name_mapping_export = source[
        source.index("function atExportNameMapping()") :
        source.index("async function atExportChain()")
    ]
    chain_export = source[
        source.index("async function atExportChain()") :
        source.index("// ─── 임베딩 설정 UI")
    ]

    # Lv2 image URLs keep the actual storage directory names.
    assert "assetNavOutfit = outfit;" in navigate
    assert "assetNavExpression = expression;" in navigate

    # Name-mapping export keys target those sanitized storage directories.
    assert "const safeName = match.name.replace(" in name_mapping_export
    assert "expressionMapping[safeName] = cleanedValue;" in name_mapping_export

    # Chain slots and per-expression settings keep the original tag name.
    assert "const exprSettings = perExpr[match.name] || {};" in chain_export
    assert "expression: match.name," in chain_export
