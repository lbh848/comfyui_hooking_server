from pathlib import Path


FRONTEND = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
    encoding="utf-8"
)


def test_visual_profile_modal_ignores_backdrop_clicks():
    modal_start = FRONTEND.index("function _vpModal()")
    modal_end = FRONTEND.index("function _vpProfile()", modal_start)
    modal_source = FRONTEND[modal_start:modal_end]

    assert "el.onclick" not in modal_source
    assert "closeVisualProfileEditor()" not in modal_source

