from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys

import numpy as np
import pytest


def test_mask_canvas_sizes_follow_the_active_bots_solo_group_and_legacy_settings():
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_illustration_mask_canvas_sizes")
    data = {"bots": [{"name": "Aster", "illust_settings_solo": {"img_w": 1600, "img_h": 900},
                     "illust_settings_group": {"img_w": 2048, "img_h": 1024}},
                    {"name": "Mira", "illust_settings": {"img_w": 900, "img_h": 1600}}]}
    namespace = {"_load_bot_data_readonly": lambda: data}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "mask_canvas_settings", "exec"), namespace)
    resolve = namespace[function.name]
    assert resolve("Aster") == {"solo": (1600, 900), "group": (2048, 1024)}
    assert resolve("Mira") == {"solo": (900, 1600), "group": (900, 1600)}
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.multi_char_mask import (
    MASK_RENDER_VERSION,
    extract_multi_char_prompt_payload,
    layout_fingerprint,
    normalize_multi_char_snapshot,
    prepare_region_mask,
    recover_multi_char_snapshot_from_sessions,
    render_region_mask,
    resolve_mask_directory,
    validate_multi_char_prompt_context,
    validate_multi_char_layout,
)


def _layout():
    return {
        "regions": [
            {"name": "Right", "x": 0.4, "y": 0.0, "width": 0.6, "height": 1.0},
            {"name": "Left", "x": 0.0, "y": 0.0, "width": 0.6, "height": 1.0},
        ]
    }


@pytest.mark.parametrize("name", ["Shoko", "Mira"])
def test_single_region_preserves_bounds_and_leaves_other_channels_and_background_empty(tmp_path, name):
    layout = validate_multi_char_layout({
        "background_prompt": "room, window light",
        "composition_prompt": "one seated figure in the center",
        "regions": [{"name": name, "x": .25, "y": .125, "width": .5, "height": .75,
                     "character_prompt": "girl, long hair, wrapped in blanket"}],
    }, [name], require_prompt_separation=True)
    assert layout["character_order"] == [name]
    assert layout["regions"][0]["channel"] == "R"
    pixels = np.asarray(render_region_mask(layout, width=64, height=32))
    assert pixels[16, 32].tolist() == [255, 0, 0]
    assert pixels[0, 0].tolist() == [0, 0, 0]
    assert not np.any(pixels[:, :, 1:])
    context = {"enable": True, "character_order": [name], "layout": layout}
    snapshot = normalize_multi_char_snapshot(context)
    assert snapshot["char_num"] == 1
    assert validate_multi_char_prompt_context(_positive_for(snapshot), snapshot) == snapshot
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    (session_dir / "sample.json").write_text(json.dumps({"items": [
        {"slot": 9, "multi_char_layout": layout}]}), encoding="utf-8")
    recovered = recover_multi_char_snapshot_from_sessions(
        str(session_dir), extract_multi_char_prompt_payload(_positive_for(snapshot)))
    assert recovered["mask_fingerprint"] == snapshot["mask_fingerprint"]
    prepared = prepare_region_mask(str(tmp_path), context)
    assert Path(prepared).is_file()
    assert not np.any(np.asarray(Image.open(prepared))[:, :, 1:])


def test_single_region_can_cover_the_actual_full_frame_crop():
    layout = validate_multi_char_layout({"regions": [
        {"name": "ClosePortrait", "x": 0, "y": 0, "width": 1, "height": 1}]}, ["ClosePortrait"])
    pixels = np.asarray(render_region_mask(layout, width=16, height=8))
    assert np.all(pixels[:, :, 0] == 255)
    assert not np.any(pixels[:, :, 1:])


@pytest.mark.parametrize("masked", [True, False])
def test_single_mask_does_not_select_group_lora_settings(masked):
    from modes.illust_prompt_builder import IllustPromptBuilder

    layout = validate_multi_char_layout({"background_prompt": "room, window light",
        "composition_prompt": "one figure with a raised hand",
        "regions": [{"name": "Shoko", "character_prompt": "girl, long hair, raised hand",
                     "x": .2, "y": .1, "width": .6, "height": .9}]}, ["Shoko"],
        require_prompt_separation=True)
    context = {"enable": True, "char_name_list": ["Shoko"],
        "char_inform": ["girl, long hair, raised hand"],
        "background_prompt": layout["background_prompt"],
        "composition_prompt": layout["composition_prompt"],
        "mask_fingerprint": layout_fingerprint(layout)} if masked else None
    bot = {"characters": [{"name": "Shoko", "gender_tag": "1girl",
        "loras_solo": [{"lora_path": "solo.safetensors", "strength": .37,
                        "trigger": "single subject", "BASE": "anima"}],
        "loras_group": [{"lora_path": "group.safetensors", "strength": .83,
                         "trigger": "group subject", "BASE": "anima"}]}]}
    result = IllustPromptBuilder().build_positive_prompt(
        "room", "Shoko, girl, long hair, raised hand", "", ["Shoko"], bot, {}, {}, "test",
        multi_char_context=context)
    payload = extract_multi_char_prompt_payload(result)
    assert payload["enable"] is masked
    assert payload["char_num"] == 1
    if masked:
        assert payload["char_trigger_list"] == [["single subject"]]
        assert payload["mask_fingerprint"] == context["mask_fingerprint"]
    loras = json.loads(result.split("[LORA_DATA]\n", 1)[1].split("\n[", 1)[0])["list"]
    assert [entry["str"] for entry in loras] == [.37]


def _separated_layout():
    layout = _layout()
    layout["background_prompt"] = "wide shot, rooftop, blue-hour city lights"
    layout["composition_prompt"] = (
        "two distinct women, one on the left holding a chart, one on the right pointing upward"
    )
    layout["regions"][0]["character_prompt"] = "black hair, purple eyes, pointing upward"
    layout["regions"][1]["character_prompt"] = "grey hair, aqua eyes, holding a star chart"
    return layout


def _snapshot():
    return normalize_multi_char_snapshot({
        "enable": True,
        "character_order": ["Left", "Right"],
        "layout": _separated_layout(),
        "mask_location": "region_mask",
    })


def _positive_for(snapshot, *, names=None, fingerprint=None):
    payload = {
        "enable": True,
        "char_num": len(names or snapshot["character_order"]),
        "char_name_list": names or snapshot["character_order"],
        "mask_fingerprint": fingerprint or snapshot["mask_fingerprint"],
    }
    return "\n".join([
        "[ANIMA_CONTENT]",
        "scene tags",
        "[MULTI_CHAR]",
        json.dumps(payload, ensure_ascii=False),
        "[HRF_ACTIVATE]",
        "false",
    ])


def test_layout_is_sorted_left_to_right_and_only_intersection_pixels_are_partitioned():
    normalized = validate_multi_char_layout(_layout(), ["Left", "Right"])

    assert normalized["character_order"] == ["Left", "Right"]
    assert [region["channel"] for region in normalized["regions"]] == ["R", "G"]

    image = render_region_mask(normalized, width=10, height=4)
    assert image.getpixel((1, 1)) == (255, 0, 0)
    assert image.getpixel((4, 1)) == (255, 0, 0)
    assert image.getpixel((5, 1)) == (0, 255, 0)
    assert image.getpixel((8, 1)) == (0, 255, 0)


def test_layout_prompt_separation_is_required_and_preserved_when_requested():
    separated = _separated_layout()

    normalized = validate_multi_char_layout(
        separated,
        ["Left", "Right"],
        require_prompt_separation=True,
    )

    assert normalized["background_prompt"] == "wide shot, rooftop, blue-hour city lights"
    assert normalized["composition_prompt"].startswith("two distinct women")
    assert [region["name"] for region in normalized["regions"]] == ["Left", "Right"]
    assert [region["character_prompt"] for region in normalized["regions"]] == [
        "grey hair, aqua eyes, holding a star chart",
        "black hair, purple eyes, pointing upward",
    ]

    with pytest.raises(ValueError, match="background_prompt"):
        validate_multi_char_layout(
            _layout(),
            ["Left", "Right"],
            require_prompt_separation=True,
        )

    background_only = _layout()
    background_only["background_prompt"] = "wide shot, rooftop"
    with pytest.raises(ValueError, match="composition_prompt"):
        validate_multi_char_layout(
            background_only,
            ["Left", "Right"],
            require_prompt_separation=True,
        )


def test_layout_accepts_spatial_only_regions_when_character_prompts_are_server_owned():
    layout = _layout()
    layout["background_prompt"] = "wide shot, rooftop, blue-hour city lights"
    layout["composition_prompt"] = "two separate figures standing side by side"

    normalized = validate_multi_char_layout(
        layout,
        ["Left", "Right"],
        require_prompt_separation=True,
        require_character_prompt=False,
    )

    assert normalized["character_order"] == ["Left", "Right"]
    assert [region["character_prompt"] for region in normalized["regions"]] == ["", ""]


def test_layout_accepts_large_overlap_without_moving_or_shrinking_body_bounds():
    layout = {
        "background_prompt": "wide shot, rooftop",
        "composition_prompt": "two separate figures",
        "regions": [
            {"name": "Left", "x": 0.0, "y": 0.0, "width": 0.8, "height": 1.0},
            {"name": "Right", "x": 0.1, "y": 0.0, "width": 0.8, "height": 1.0},
        ],
    }

    normalized = validate_multi_char_layout(layout, ["Left", "Right"],
        require_prompt_separation=True, require_character_prompt=False)
    for actual, original in zip(normalized["regions"], layout["regions"]):
        assert {key: actual[key] for key in ("x", "y", "width", "height")} == {
            key: original[key] for key in ("x", "y", "width", "height")}
    pixels = np.asarray(render_region_mask(normalized, width=100, height=80))
    assert not np.any(np.count_nonzero(pixels, axis=2) > 1)
    assert np.any(pixels[:, :, 0]) and np.any(pixels[:, :, 1])


@pytest.mark.parametrize("names", [("Doyun", "Shoko"), ("Nora", "Ilan")])
@pytest.mark.parametrize("transform", ["original", "mirror_x", "mirror_y", "transpose"])
@pytest.mark.parametrize("size", [(400, 240), (240, 400)])
def test_unequal_body_overlap_is_split_at_its_midpoint(names, transform, size):
    # The supplied PNG's actual layout: a smaller background body overlaps
    # the left edge of a larger foreground body. V3 gave all of it to the former.
    bounds = [(.1, .25, .3, .6), (.25, .2, .5, .7)]
    regions = []
    for name, (x, y, w, h) in zip(names, bounds):
        if transform == "mirror_x":
            x = round(1 - x - w, 12)
        elif transform == "mirror_y":
            y = round(1 - y - h, 12)
        elif transform == "transpose":
            x, y, w, h = y, x, h, w
        regions.append(dict(name=name, x=x, y=y, width=w, height=h))
    layout = validate_multi_char_layout({"regions": regions}, names)
    original = deepcopy(layout)
    before = np.asarray(render_region_mask({**layout, "mask_render_version": 1}, *size)) != 0
    after = np.asarray(render_region_mask(layout, *size)) != 0
    rows, columns = np.nonzero(before[:, :, 0] & before[:, :, 1])
    axis = rows if transform == "transpose" else columns
    midpoint = (axis.min() + axis.max() + 1) / 2
    smaller_side = axis + .5 < midpoint
    if transform == "mirror_x":
        smaller_side = ~smaller_side
    smaller_channel = layout["character_order"].index(names[0])
    larger_channel = layout["character_order"].index(names[1])
    assert np.array_equal(after[rows, columns, smaller_channel], smaller_side)
    assert np.array_equal(after[rows, columns, larger_channel], ~smaller_side)
    assert after[rows, columns, smaller_channel].sum() == len(rows) // 2
    assert after[rows, columns, larger_channel].sum() == len(rows) // 2
    exclusive = before.sum(axis=2) == 1
    assert np.array_equal(after[exclusive], before[exclusive])
    assert layout == original


@pytest.mark.parametrize("bounds", [
    [(0.25, 0.2, 0.5, 0.6), (0.2, 0.4, 0.6, 0.5)],
    [(0.1, 0.15, 0.65, 0.75), (0.25, 0.05, 0.65, 0.75)],
    [(0.05, 0.05, 0.9, 0.9), (0.35, 0.35, 0.3, 0.3)],
    [(0.05, 0.05, 0.75, 0.8), (0.2, 0.2, 0.65, 0.65), (0.35, 0.35, 0.6, 0.6)],
    [(0.1, 0.1, 0.8, 0.8)] * 2,
    [(0.1, 0.1, 0.8, 0.8)] * 3,
    [(0.25, 0.05, 0.5, 0.9)] * 3,
    # Distinct concentric rectangles and duplicate centers beside a third body.
    [(0.1, 0.1, 0.8, 0.8), (0.2, 0.2, 0.6, 0.6), (0.35, 0.35, 0.3, 0.3)],
    [(0.1, 0.1, 0.6, 0.8)] * 2 + [(0.3, 0.2, 0.6, 0.6)],
    [(0.05, 0.1, 0.8, 0.8), (0.475, 0.4, 0.05, 0.2), (0.15, 0.1, 0.8, 0.8)],
    # Opposite cases: disjoint, touching and a single body need no partition.
    [(0.0, 0.0, 0.3, 1.0), (0.7, 0.0, 0.3, 1.0)],
    [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)],
    [(0.2, 0.1, 0.5, 0.25), (0.3, 0.6, 0.5, 0.3)],
    [(0.25, 0.2, 0.5, 0.6)],
])
@pytest.mark.parametrize("version", [4, 5])
def test_midpoint_partition_preserves_coverage_ownership_and_all_channels(bounds, version):
    names = [f"Subject-{index}" for index in range(len(bounds))]
    layout = validate_multi_char_layout({"mask_render_version": version, "regions": [
        dict(name=name, x=x, y=y, width=w, height=h)
        for name, (x, y, w, h) in zip(names, bounds)
    ]}, names)
    original = deepcopy(layout)
    before = np.asarray(render_region_mask({**layout, "mask_render_version": 1}, 180, 120)) != 0
    expanded = deepcopy(layout)
    expanded["mask_render_version"] = 1
    if version == 4:
        for region in expanded["regions"]:
            region["height"] += region["y"]
            region["y"] = 0
    allowed = np.asarray(render_region_mask(expanded, 180, 120)) != 0
    image = render_region_mask(layout, 180, 120)
    after = np.asarray(image) != 0
    assert np.array_equal(after.any(axis=2), allowed.any(axis=2))
    exclusive = before.sum(axis=2) == 1
    assert np.array_equal(after[exclusive], before[exclusive])
    assert not np.any(after & ~allowed)
    assert not np.any(after.sum(axis=2) > 1)
    assert all(after[:, :, index].any() for index in range(len(bounds)))
    assert image.tobytes() == render_region_mask(layout, 180, 120).tobytes()
    assert layout == original


@pytest.mark.parametrize("version,expected", [
    (1, "76a584c3783a078d54332608976e2ab8f2b9ce49db169e31e81114063822de03"),
    (2, "6ebf3fe83979e326272eff86eb57fe39edb0a031e87689fc50b3fd4b66b87f9a"),
    (3, "0908f9a94a77a5d0dfb95a42740d46751eaf546de558540a30f80d6ddbe7e1be"),
])
def test_saved_masks_keep_their_pre_upgrade_pixels(version, expected):
    layout = validate_multi_char_layout({"mask_render_version": version, "regions": [
        dict(name="Doyun", x=.1, y=.25, width=.3, height=.6),
        dict(name="Shoko", x=.25, y=.2, width=.5, height=.7),
    ]}, ["Doyun", "Shoko"])
    snapshot = normalize_multi_char_snapshot({"enable": True, "layout": layout,
                                             "character_order": layout["character_order"]})
    image = render_region_mask(snapshot["layout"], 100, 100)
    assert hashlib.sha256(image.tobytes()).hexdigest() == expected
    assert normalize_multi_char_snapshot(snapshot) == snapshot
    assert snapshot["mask_fingerprint"] != layout_fingerprint({**layout, "mask_render_version": MASK_RENDER_VERSION})


@pytest.mark.parametrize("names", [("Doyun", "Shoko", "Hibiki"), ("Nora", "Ilan", "Cleo")])
@pytest.mark.parametrize("bounds", [
    # Actual failed response: 66.7% overlap of the smaller rectangle.
    [(0.25, 0.2, 0.5, 0.6), (0.2, 0.4, 0.6, 0.5)],
    [(0.1, 0.15, 0.65, 0.75), (0.25, 0.05, 0.65, 0.75)],
    [(0.05, 0.05, 0.9, 0.9), (0.35, 0.35, 0.3, 0.3)],
    [(0.05, 0.05, 0.75, 0.8), (0.2, 0.2, 0.65, 0.65), (0.35, 0.35, 0.6, 0.6)],
    [(0.1, 0.1, 0.8, 0.8)] * 2,
    [(0.1, 0.1, 0.8, 0.8)] * 3,
    [(0.25, 0.05, 0.5, 0.9)] * 3,
    [(0.0, 0.0, 0.3, 1.0), (0.7, 0.0, 0.3, 1.0)],
    [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)],
])
def test_partition_preserves_union_exclusive_pixels_identity_and_determinism(names, bounds):
    layout = {"regions": [dict(name=name, x=x, y=y, width=w, height=h)
        for name, (x, y, w, h) in zip(names, bounds)]}
    original = deepcopy(layout)
    normalized = validate_multi_char_layout(layout, names[:len(bounds)])
    normalized["mask_render_version"] = 2
    legacy = {**normalized, "mask_render_version": 1}
    before = np.asarray(render_region_mask(legacy, width=180, height=120)) != 0
    image = render_region_mask(normalized, width=180, height=120)
    after = np.asarray(image) != 0
    assert layout == original
    assert image.tobytes() == render_region_mask(normalized, width=180, height=120).tobytes()
    assert np.array_equal(after.any(axis=2), before.any(axis=2))
    exclusive = before.sum(axis=2) == 1
    assert np.array_equal(after[exclusive], before[exclusive])
    assert not np.any(after.sum(axis=2) > 1)
    assert all(after[:, :, index].any() for index in range(len(bounds)))
    assert not np.any(after & ~before)


def test_diagonal_overlap_boundary_follows_positions_in_both_axes():
    layout = validate_multi_char_layout({"regions": [
        {"name": "Upper", "x": 0, "y": 0, "width": 0.8, "height": 0.8},
        {"name": "Lower", "x": 0.2, "y": 0.2, "width": 0.8, "height": 0.8},
    ]}, ["Upper", "Lower"])
    image = render_region_mask(layout, width=100, height=100)
    assert image.getpixel((60, 30)) == (255, 0, 0)
    assert image.getpixel((60, 70)) == (0, 255, 0)
    assert image.getpixel((30, 60)) == (255, 0, 0)
    assert image.getpixel((70, 60)) == (0, 255, 0)


def test_legacy_snapshot_fingerprint_and_mask_survive_new_render_algorithm():
    legacy = _snapshot()
    fresh = normalize_multi_char_snapshot({"enable": True,
        "character_order": ["Left", "Right"],
        "layout": validate_multi_char_layout(_separated_layout(), ["Left", "Right"])})
    assert legacy["layout"]["mask_render_version"] == 1
    assert fresh["layout"]["mask_render_version"] == MASK_RENDER_VERSION
    assert legacy["mask_fingerprint"] != fresh["mask_fingerprint"]
    assert render_region_mask(legacy["layout"], width=10, height=4).getpixel((5, 1)) == (255, 255, 0)
    assert render_region_mask(fresh["layout"], width=10, height=4).getpixel((5, 1)) == (0, 255, 0)
    stored_before_upgrade = deepcopy(legacy)
    stored_before_upgrade["layout"].pop("mask_render_version")
    assert normalize_multi_char_snapshot(stored_before_upgrade) == legacy
    assert normalize_multi_char_snapshot(legacy) == legacy
    assert normalize_multi_char_snapshot(fresh) == fresh


@pytest.mark.parametrize("fresh", [False, True])
def test_installed_mask_preparation_uses_the_saved_render_version(tmp_path, fresh):
    layout = validate_multi_char_layout(_layout(), ["Left", "Right"]) if fresh else _layout()
    snapshot = normalize_multi_char_snapshot({"enable": True,
        "character_order": ["Left", "Right"], "layout": layout})
    path = prepare_region_mask(str(tmp_path), snapshot)
    with Image.open(path) as image:
        pixels = np.asarray(image)
        assert bool(np.any(np.count_nonzero(pixels, axis=2) > 1)) is not fresh
        assert image.getpixel((50, 512)) == (255, 0, 0)
        assert image.getpixel((950, 512)) == (0, 255, 0)


def test_layout_fingerprint_changes_when_only_mask_coordinates_change():
    first = validate_multi_char_layout(_layout(), ["Left", "Right"])
    changed_layout = _layout()
    changed_layout["regions"][1]["width"] = 0.5
    second = validate_multi_char_layout(changed_layout, ["Left", "Right"])

    assert len(layout_fingerprint(first)) == 64
    assert layout_fingerprint(first) != layout_fingerprint(second)


@pytest.mark.parametrize("prefix", ["Original", "Renamed"])
@pytest.mark.parametrize("bounds", [
    [(0.05, 0.3, 0.45, 0.7), (0.6, 0.15, 0.3, 0.75)],
    [(0.65, 0.25, 0.3, 0.6)],
    [(0.32, 0.21, 0.36, 0.68)],
    [(0.25, 0.3, 0.5, 0.6)],
    # Reaching subjects and head-off-frame crops retain their original width/bottom.
    [(0.1, 0.2, 0.8, 0.3)],
    [(0.2, 0, 0.5, 0.7)],
    # A lower subject's upward extension must not take the upper subject's body.
    [(0.2, 0.1, 0.5, 0.25), (0.3, 0.6, 0.5, 0.3)],
])
def test_upward_extension_preserves_original_ownership_and_only_adds_above(prefix, bounds):
    regions = [dict(name=f"{prefix}-{index}", x=x, y=y, width=w, height=h)
               for index, (x, y, w, h) in enumerate(bounds)]
    layout = validate_multi_char_layout({"mask_render_version": 3, "regions": regions},
                                      [r["name"] for r in regions])
    original = deepcopy(layout)
    previous = {**layout, "mask_render_version": 2}
    before = np.asarray(render_region_mask(previous, width=200, height=100))
    after = np.asarray(render_region_mask(layout, width=200, height=100))
    occupied = before.any(axis=2)
    assert np.array_equal(after[occupied], before[occupied])
    assert not np.any(np.count_nonzero(after, axis=2) > 1)
    for index, (x, y, width, _height) in enumerate(bounds):
        added_y, added_x = np.nonzero((after[:, :, index] > 0) & (before[:, :, index] == 0))
        assert np.all(added_y < int(np.floor(y * 100)))
        assert np.all(added_x >= int(np.floor(x * 200)))
        assert np.all(added_x < int(np.ceil((x + width) * 200)))
    assert layout == original
    assert after.tobytes() == render_region_mask(layout, width=200, height=100).tobytes()
    assert layout_fingerprint(previous) != layout_fingerprint(layout)
    if len(bounds) == 1:
        left = int(np.floor(bounds[0][0] * 200))
        right = int(np.ceil((bounds[0][0] + bounds[0][2]) * 200))
        assert np.all(after[0, left:right, 0] == 255)
    if all(y == 0 for _x, y, _w, _h in bounds):
        assert np.array_equal(before, after)


@pytest.mark.parametrize("version", [1, 2])
def test_saved_pre_extension_masks_keep_their_original_top(version):
    layout = validate_multi_char_layout({"mask_render_version": version, "regions": [
        {"name": "Portrait", "x": .25, "y": .3, "width": .5, "height": .6},
    ]}, ["Portrait"])
    snapshot = normalize_multi_char_snapshot({"enable": True, "layout": layout,
                                              "character_order": ["Portrait"]})
    image = render_region_mask(snapshot["layout"], width=100, height=100)
    assert image.getpixel((50, 29)) == (0, 0, 0)
    assert image.getpixel((50, 30)) == (255, 0, 0)
    assert normalize_multi_char_snapshot(snapshot) == snapshot


def test_prepare_region_mask_clears_folder_only_when_called(tmp_path):
    input_dir = tmp_path / "input"
    mask_dir = input_dir / "region_mask"
    stale_dir = mask_dir / "old-folder"
    stale_dir.mkdir(parents=True)
    (mask_dir / "old.png").write_bytes(b"old")
    (stale_dir / "old.txt").write_text("old", encoding="utf-8")

    # 큐 적재 단계에는 helper를 호출하지 않으므로 기존 파일이 그대로 있다.
    assert (mask_dir / "old.png").exists()

    final_path = prepare_region_mask(
        str(input_dir),
        {
            "enable": True,
            "characters": [{"name": "Left"}, {"name": "Right"}],
            "character_order": ["Left", "Right"],
            "layout": _layout(),
            "mask_location": "region_mask",
        },
    )

    assert Path(final_path).name == "region_mask.png"
    assert [path.name for path in mask_dir.iterdir()] == ["region_mask.png"]
    with Image.open(final_path) as image:
        assert image.mode == "RGB"
        assert image.size == (1024, 1024)


def test_mask_location_cannot_escape_input_directory(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="밖"):
        resolve_mask_directory(str(input_dir), "../outside")

    with pytest.raises(ValueError, match="비어"):
        resolve_mask_directory("", "region_mask")


def test_multi_char_snapshot_and_prompt_control_block_must_match():
    snapshot = _snapshot()
    positive = _positive_for(snapshot)

    payload = extract_multi_char_prompt_payload(positive)
    validated = validate_multi_char_prompt_context(positive, snapshot)

    assert payload["char_name_list"] == ["Left", "Right"]
    assert validated == snapshot
    assert validated["layout"]["character_order"] == ["Left", "Right"]

    with pytest.raises(ValueError, match="캐릭터 순서"):
        validate_multi_char_prompt_context(
            _positive_for(snapshot, names=["Right", "Left"]),
            snapshot,
        )
    with pytest.raises(ValueError, match="지문"):
        validate_multi_char_prompt_context(
            _positive_for(snapshot, fingerprint="0" * 64),
            snapshot,
        )


def test_multi_char_snapshot_rejects_changed_coordinates_with_old_fingerprint():
    snapshot = _snapshot()
    changed = _separated_layout()
    changed["regions"][1]["width"] = 0.5

    with pytest.raises(ValueError, match="지문"):
        normalize_multi_char_snapshot({
            "enable": True,
            "character_order": ["Left", "Right"],
            "layout": changed,
            "mask_fingerprint": snapshot["mask_fingerprint"],
        })


def test_three_character_snapshot_preserves_rgb_order():
    layout = {
        "regions": [
            {"name": "Center", "x": 0.3, "y": 0.0, "width": 0.4, "height": 1.0},
            {"name": "Right", "x": 0.65, "y": 0.0, "width": 0.35, "height": 1.0},
            {"name": "Left", "x": 0.0, "y": 0.0, "width": 0.35, "height": 1.0},
        ]
    }
    snapshot = normalize_multi_char_snapshot({
        "enable": True,
        "character_order": ["Left", "Center", "Right"],
        "layout": layout,
    })
    positive = _positive_for(snapshot)

    validated = validate_multi_char_prompt_context(positive, snapshot)
    image = render_region_mask(validated["layout"], width=12, height=3)

    assert validated["character_order"] == ["Left", "Center", "Right"]
    assert [region["channel"] for region in validated["layout"]["regions"]] == [
        "R",
        "G",
        "B",
    ]
    assert image.getpixel((1, 1)) == (255, 0, 0)
    assert image.getpixel((6, 1))[1] == 255
    assert image.getpixel((10, 1)) == (0, 0, 255)


def test_legacy_snapshot_is_recovered_only_by_matching_session_fingerprint(tmp_path):
    snapshot = _snapshot()
    payload = extract_multi_char_prompt_payload(_positive_for(snapshot))
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    (session_dir / "risu_test.json").write_text(
        json.dumps({
            "items": [
                {"slot": 1, "multi_char_layout": _separated_layout()},
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    recovered = recover_multi_char_snapshot_from_sessions(
        str(session_dir),
        payload,
    )

    assert recovered["mask_fingerprint"] == snapshot["mask_fingerprint"]
    assert recovered["character_order"] == ["Left", "Right"]

    bad_payload = dict(payload)
    bad_payload["mask_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        recover_multi_char_snapshot_from_sessions(str(session_dir), bad_payload)
