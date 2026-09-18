from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _css_rule(selector: str) -> str:
    return FRONTEND.split(f"{selector} {{", 1)[1].split("}", 1)[0]


def test_sysmon_separates_host_and_gpu_stats_into_two_columns() -> None:
    strip_rule = _css_rule(".sysmon-strip")

    assert "display: inline-grid;" in strip_rule
    assert "grid-template-columns: auto auto;" in strip_rule
    assert 'class="sysmon-strip" tabindex="0"' in FRONTEND
    assert '<div class="sysmon-host-rows">' in FRONTEND
    assert '<div id="sysmon-gpu-rows" class="sysmon-gpu-rows"></div>' in FRONTEND


def test_sysmon_adds_two_rows_for_each_detected_gpu() -> None:
    assert "while (container.children.length < want * 2)" in FRONTEND
    assert "const gpuRow = container.children[i * 2];" in FRONTEND
    assert "const vramRow = container.children[i * 2 + 1];" in FRONTEND


def test_sysmon_has_compact_summary_and_hover_detail_states() -> None:
    assert ".sysmon-strip:hover," in FRONTEND
    assert ".sysmon-strip:focus-visible" in FRONTEND
    assert (
        ".sysmon-strip:not(:hover):not(:focus-visible) "
        ".sysmon-host-rows .sysmon-row:nth-child(n + 2)"
    ) in FRONTEND
    assert (
        ".sysmon-strip:not(:hover):not(:focus-visible) "
        ".sysmon-gpu-rows .sysmon-row:nth-child(even)"
    ) in FRONTEND


def test_header_controls_are_a_retractable_top_layer_dock() -> None:
    header_rule = _css_rule("header.app-header")
    controls_rule = _css_rule("header.app-header .controls")

    assert "isolation: auto;" in header_rule
    assert "position: fixed;" in controls_rule
    assert "z-index: 2147483647;" in controls_rule
    assert "transform: translateY(calc(-100% + var(--header-dock-peek)));" in controls_rule
    assert "header.app-header .controls.is-pinned" in FRONTEND
    assert "header.app-header .controls.is-hover-active" not in FRONTEND
    assert "header.app-header .controls:has(:focus-visible)" not in FRONTEND
