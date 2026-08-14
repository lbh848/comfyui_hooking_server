from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_prompt_format_label_is_derived_from_selected_workflow_profile():
    source = _frontend_source()
    label_function = _function_source(
        source, "_currentPromptFormatLabel()", "updateBotModeStatus(checked)"
    )

    assert "normalizeIllustrationWorkflowType" in label_function
    assert "illustrationPromptFormatForType(profile)" in label_function
    assert "setting-illustration-provider-chansub" not in label_function


def test_prompt_format_is_auto_selected_and_read_only():
    source = _frontend_source()
    lock_function = _function_source(
        source, "_applyPromptFormatChansubLock()", "_applyCall3PromptModeLock()"
    )

    assert "illustrationPromptFormatForType(profile)" in lock_function
    assert "_illustrationContextToggles.prompt_format = target;" in lock_function
    assert "select.disabled = true;" in lock_function
    assert "fetch(" not in lock_function
    assert "'/api/config'" not in lock_function


def test_chansub_settings_expose_bounded_concurrency_and_rotation_key():
    source = _frontend_source()

    assert 'id="setting-chansub-max-concurrency" min="1" max="2"' in source
    assert "currentConfig.chansub_max_concurrency ?? 1" in source
    assert "chansub_max_concurrency: parseInt(" in source
    assert 'id="setting-chansub-rotation-api-key"' in source
    assert "rotation_api_key: rotationApiKey" in source


def test_chansub_settings_expose_builtin_quality_tag_filter():
    source = _frontend_source()

    assert 'id="setting-chansub-strip-builtin-quality-tags"' in source
    assert "currentConfig.chansub_strip_builtin_quality_tags !== false" in source
    assert (
        "chansub_strip_builtin_quality_tags: "
        "document.getElementById('setting-chansub-strip-builtin-quality-tags').checked"
    ) in source
    assert "저장·백업 프롬프트는 원문을 유지합니다." in source


def test_queue_groups_modal_and_vast_in_one_color_coded_cloud_lane():
    source = _frontend_source()

    assert "{key: 'hybrid', icon: '⚡', title: '동적 배분'" in source
    assert "먼저 빈 로컬 GPU/Modal/Vast/챈섭이 실행" in source
    assert "key: 'cloud', icon: '☁️', title: '클라우드 GPU'" in source
    assert "itemAreas: ['modal', 'vast']" in source
    assert "{key: 'modal', label: 'Modal'}" in source
    assert "{key: 'vast', label: 'Vast'}" in source
    assert "_queueDefinitionIncludesArea(def, _queueItemArea(item))" in source
    assert 'data-queue-area="cloud"' in source
    assert ".filter(child => child.dataset.queueArea === area);" in source
    assert "if (area === 'comfy_parallel') return 'hybrid';" in source
    assert "grid-template-columns:repeat(6, minmax(0, 1fr))" in source
    assert ".queue-lane.hybrid { border-top:2px solid #f59e0b; }" in source
    assert ".queue-lane.cloud { border-top:2px solid #60a5fa; }" in source
    assert '.queue-cloud-key[data-provider="modal"] .queue-cloud-dot' in source
    assert '.queue-cloud-key[data-provider="vast"] .queue-cloud-dot' in source
    assert '.queue-modal-item[data-queue-area="modal"]' in source
    assert '.queue-modal-item[data-queue-area="vast"]' in source


def test_backup_card_has_modal_and_chansub_source_badges():
    """삽화 백업 카드에 Modal=M / 챈섭=S 딱지가 썸네일 상단에 오버레이로 표시된다.
    로컬 GPU 백업은 배지 없음. 라이트박스는 img.src만 확대하므로 배지가 보이지 않는다."""
    source = _frontend_source()
    render = _function_source(source, "renderBackups(backups) {", "renderConversionInfoContent(info)")

    # CSS: 썸네일 좌상단 고정, Modal/챈섭 색상 분기.
    assert ".card-image-area .source-badge {" in source
    assert "position: absolute;" in source
    assert ".card-image-area .source-badge.source-modal {" in source
    assert ".card-image-area .source-badge.source-chansub {" in source

    # 렌더: execution_source 기반 M/S 분기. 구 백업은 provider fallback.
    # 전역 다이얼로그 .modal과 충돌하지 않도록 source- 접두 클래스를 사용한다.
    assert "info.execution_source" in render
    assert "source-badge source-${sourceKind}" in render
    assert 'class="source-badge ${sourceKind}"' not in render
    assert "${sourceKind === 'modal' ? 'M' : 'S'}" in render
    # 라이트박스는 배지 없이 img.src만 확대 — openLightbox가 source-badge를 참조하지 않는다.
    lightbox = _function_source(source, "openLightbox(src) {", "closeLightbox()")
    assert "source-badge" not in lightbox


def test_queue_full_item_names_are_available_on_hover():
    source = _frontend_source()

    assert "if (batchLabelEl) batchLabelEl.title = label;" in source
    assert "if (labelEl) labelEl.title = item.label || '';" in source
    assert "if (labelEl) labelEl.title = subtask.label || '하위 작업';" in source
