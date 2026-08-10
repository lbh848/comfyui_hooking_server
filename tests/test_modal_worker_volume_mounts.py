from __future__ import annotations

from pathlib import Path


def test_modal_worker_does_not_reload_volumes_while_comfyui_is_running() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "modal_backend" / "modal_app.py"
    ).read_text(encoding="utf-8")

    assert "volume_reload" not in source
    assert "models_volume.reload()" not in source
    assert "loras_volume.reload()" not in source
    assert "reload_volume_with_retry" not in source
