from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_asset_gallery_sort_dropdown_has_all_requested_modes():
    controls = FRONTEND[
        FRONTEND.index("function buildAssetGalleryControls(scope)"):
        FRONTEND.index("function restoreAssetGallerySearchValue(scope)")
    ]

    assert '<option value="representative_missing"' in controls
    assert ">대표 이미지 없음</option>" in controls
    assert '<option value="representative_present"' in controls
    assert ">대표 이미지 있음</option>" in controls
    assert '<option value="oldest"' in controls
    assert ">과거순</option>" in controls
    assert '<option value="newest"' in controls
    assert ">최신순</option>" in controls
    assert '<option value="alphabetical"' in controls
    assert ">알파벳순</option>" in controls
    assert 'type="search"' in controls
    assert "filterAssetGallery('${scope}', this.value)" in controls


def test_asset_and_upload_breadcrumbs_use_the_shared_gallery_controls():
    upload_breadcrumb = FRONTEND[
        FRONTEND.index("function auUpdateBreadcrumb()"):
        FRONTEND.index("async function auNavigateBreadcrumb(level)")
    ]
    asset_breadcrumb = FRONTEND[
        FRONTEND.index("function updateAssetBreadcrumb()"):
        FRONTEND.index("function updateGenerateBtn()")
    ]

    assert "auNavLevel === 1 && auNavCharacter" in upload_breadcrumb
    assert "buildAssetGalleryControls('upload')" in upload_breadcrumb
    assert "restoreAssetGallerySearchValue('upload')" in upload_breadcrumb
    assert "assetNavLevel === 1 && assetNavCharacter" in asset_breadcrumb
    assert "buildAssetGalleryControls('asset')" in asset_breadcrumb
    assert "restoreAssetGallerySearchValue('asset')" in asset_breadcrumb


def test_gallery_cards_include_browser_sort_metadata_and_apply_controls():
    upload_gallery = FRONTEND[
        FRONTEND.index("async function auRenderGallery(charName)"):
        FRONTEND.index("async function auNavigateToImages(charName, outfit, expression)")
    ]
    asset_gallery = FRONTEND[
        FRONTEND.index("async function renderCharacterGallery(charName)"):
        FRONTEND.index("let _assetScrollCache = {}")
    ]

    for gallery, scope in ((upload_gallery, "upload"), (asset_gallery, "asset")):
        assert "card.dataset.hasRepresentative" in gallery
        assert "card.dataset.modifiedAt" in gallery
        assert f"applyAssetGalleryControls('{scope}')" in gallery


def test_browser_sorting_keeps_asset_groups_and_filters_by_outfit_and_expression():
    apply_controls = FRONTEND[
        FRONTEND.index("function applyAssetGalleryControls(scope)"):
        FRONTEND.index("function auShowView(view)")
    ]

    assert "card.dataset.outfit" in apply_controls
    assert "card.dataset.expression" in apply_controls
    assert "searchableName.includes(normalizedQuery)" in apply_controls
    assert "gallery-group-container" in apply_controls
    assert "groupCards.sort" in apply_controls
    assert "group.hidden = !groupCards.some" in apply_controls
