import json
from pathlib import Path

from modes.asset_mode import AssetMode


ROOT = Path(__file__).resolve().parents[1]


def test_negative_metadata_sync_updates_edit_override_and_preserves_fields(tmp_path):
    prompt_path = tmp_path / "edited_prompt.json"
    prompt_path.write_text(
        json.dumps(
            {
                "positive": "1girl, blue hair",
                "negative": "old base negative",
                "is_edited": True,
                "edit_prompt": "change the jacket",
                "edit_negative_prompt": "old edit negative",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    updated = AssetMode.update_prompt_negative_metadata(
        str(prompt_path),
        "new shared negative",
    )

    saved = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert updated == saved
    assert saved["negative"] == "new shared negative"
    assert saved["edit_negative_prompt"] == "new shared negative"
    assert saved["positive"] == "1girl, blue hair"
    assert saved["edit_prompt"] == "change the jacket"


def test_negative_metadata_sync_keeps_unedited_asset_without_edit_override(tmp_path):
    prompt_path = tmp_path / "uploaded_prompt.json"
    prompt_path.write_text(
        json.dumps({"positive": "uploaded asset"}, ensure_ascii=False),
        encoding="utf-8",
    )

    AssetMode.update_prompt_negative_metadata(
        str(prompt_path),
        "uploaded negative",
    )

    saved = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert saved["negative"] == "uploaded negative"
    assert "edit_negative_prompt" not in saved


def test_edit_modal_refreshes_latest_metadata_for_upload_and_generation_views():
    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    helper = frontend[
        frontend.index("async function auLoadLatestEditSourceData(data)"):
        frontend.index("async function auOpenQwenEdit(data)")
    ]
    opener = frontend[
        frontend.index("async function auOpenQwenEdit(data)"):
        frontend.index("function auCloseQwenEdit()")
    ]
    refresh = frontend[
        frontend.index("async function auRefreshPromptViews("):
        frontend.index("function auUpdateBreadcrumb()")
    ]
    upload_negative = frontend[
        frontend.index("async function auBatchSetNegative()"):
        frontend.index("async function auBatchSetNegativeSelected()")
    ]
    selected_negative = frontend[
        frontend.index("async function auBatchSetNegativeSelected()"):
        frontend.index("async function auRefreshPromptViews(")
    ]
    generation_view = frontend[
        frontend.index("async function loadAssetImages()"):
        frontend.index("async function setAssetRepresentative(")
    ]

    assert "fetch(metadataUrl, { cache: 'no-store' })" in helper
    assert "latest.edit_negative_prompt || latest.negative || ''" in helper
    assert "await auWaitForPendingNegativePromptApplies()" in opener
    assert "data = await auLoadLatestEditSourceData(data)" in opener
    assert "openRequestId !== auQwenEditOpenRequestId" in opener
    assert "const auPendingNegativePromptApplyRequests = new Set()" in frontend
    assert "Promise.allSettled(pending)" in frontend
    assert "await auRefreshPromptViews(targetCharacter)" in upload_negative
    assert "await auRefreshPromptViews(" in selected_negative
    assert "auRenderImages(auNavCharacter, auNavOutfit, auNavExpression)" in refresh
    assert "refreshes.push(loadAssetImages())" in refresh
    assert "auOpenQwenEdit({" in generation_view


def test_batch_negative_route_uses_shared_metadata_sync():
    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    handler = server_source[
        server_source.index("async def handle_api_asset_mode_batch_set_negative("):
        server_source.index("def _character_maker_error_response(")
    ]

    assert "asset_mode.update_prompt_negative_metadata(" in handler
    assert "except Exception:\n                        pass" not in handler
