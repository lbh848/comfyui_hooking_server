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
    assert '<div class="sysmon-host-rows">' in FRONTEND
    assert '<div id="sysmon-gpu-rows" class="sysmon-gpu-rows"></div>' in FRONTEND


def test_sysmon_adds_two_rows_for_each_detected_gpu() -> None:
    assert "while (container.children.length < want * 2)" in FRONTEND
    assert "const gpuRow = container.children[i * 2];" in FRONTEND
    assert "const vramRow = container.children[i * 2 + 1];" in FRONTEND

