"""설정 폼이 채워지기 전에 저장되어 설정이 통째로 날아가는 회귀를 막는다.

saveSettings() 는 설정 모달의 DOM 을 통째로 읽어 /api/config 로 보낸다.
그런데 폼을 채우는 곳은 populateSettingsForm() 뿐이고, 모달을 열지 않고
저장하는 경로가 존재한다(Modal 런타임 패널의 "설정 저장·적용" 버튼이
saveSettings() 를 직접 부른다).

폼이 비어 있으면 HTML 기본값이 저장된다. 실측으로 28개 키가 덮였고,
그 안에는 comfy_task_allocations(원격 배분 → 전부 로컬 1),
workflow_base_dir(경로 → ""), illustration_workflow_source_paths,
modal_enabled, LLM 설정이 포함됐다. 즉 설정이 통째로 초기화된다.
"""

from __future__ import annotations

from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _slice_function(source: str, header: str) -> str:
    """`header` 로 시작하는 최상위 함수 본문을 닫는 중괄호까지 잘라낸다."""
    start = source.index(header)
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"함수 끝을 찾지 못했습니다: {header}")


def test_form_population_is_separate_from_modal_display() -> None:
    """폼 채우기가 모달 표시와 분리돼 있어야 저장 경로에서 재사용할 수 있다."""
    assert "async function populateSettingsForm()" in FRONTEND
    assert "async function openSettingsModal()" in FRONTEND

    open_modal = _slice_function(FRONTEND, "async function openSettingsModal()")
    # 모달 열기는 반드시 폼 채우기를 거친다.
    assert "await populateSettingsForm()" in open_modal
    assert "settings-modal" in open_modal and "classList.add('visible')" in open_modal

    populate = _slice_function(FRONTEND, "async function populateSettingsForm()")
    # 폼 채우기 자체는 모달을 띄우지 않는다(저장 경로에서 쓰이므로).
    assert "classList.add('visible')" not in populate
    # 채움 완료 표시를 남긴다.
    assert "settingsFormPopulated = true" in populate


def test_save_settings_refuses_to_write_an_unpopulated_form() -> None:
    """saveSettings() 는 폼이 비어 있으면 먼저 채운 뒤에 저장해야 한다."""
    save = _slice_function(FRONTEND, "async function saveSettings()")
    assert "if (!settingsFormPopulated)" in save
    assert "await populateSettingsForm()" in save

    # 가드가 폼을 읽기 시작하는 지점보다 앞에 있어야 의미가 있다.
    guard_at = save.index("if (!settingsFormPopulated)")
    first_read_at = save.index("document.getElementById")
    assert guard_at < first_read_at, (
        "설정 폼을 읽기 전에 채움 여부를 확인해야 합니다."
    )


def test_populated_flag_defaults_to_false() -> None:
    """플래그 기본값이 true 면 가드가 무력화된다."""
    assert "let settingsFormPopulated = false;" in FRONTEND


def test_config_post_sites_carry_an_origin_hint() -> None:
    """설정 되돌림 추적을 위해 모든 /api/config POST 는 _origin 을 붙인다."""
    post_sites = FRONTEND.count("'/api/config'")
    origin_hints = FRONTEND.count("_origin: '")
    # GET 조회 2곳을 제외한 나머지가 POST 저장 지점이다.
    assert origin_hints >= post_sites - 2, (
        f"_origin 힌트가 부족합니다: sites={post_sites}, hints={origin_hints}"
    )
