"""Dynamic mask geometry and STEP6 delivery, without production data writes."""
import asyncio
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from modes import multi_char_mask
import illustration_flow

NODE_DIR = Path(__file__).resolve().parents[1] / "comfy/custom_nodes/comfyui-soya-custom-nodes"
spec = importlib.util.spec_from_file_location("_face_tracking_tests", NODE_DIR / "soya_face_tracking.py")
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)
FORMAT = SimpleNamespace(latent_rgb_factors=np.zeros((16, 3)),
                         latent_rgb_factors_bias=[0, 0, 0], spacial_downscale_ratio=8)


@pytest.mark.parametrize("version", [None, 3, 4, 5])
def test_generator_control_preserves_mask_version_without_changing_character_prompt(version):
    from modes.illust_prompt_builder import IllustPromptBuilder
    context = dict(enable=True, char_name_list=["Mira"], char_inform=["girl, silver hair"],
                   background_prompt="room", composition_prompt="one seated figure",
                   mask_fingerprint="a" * 64)
    if version is not None:
        context["mask_render_version"] = version
    result = IllustPromptBuilder().build_positive_prompt(
        "room", "Mira, girl, silver hair", "", ["Mira"],
        {"characters": [{"name": "Mira", "gender_tag": "1girl"}]}, {}, {}, "test",
        multi_char_context=context)
    payload = multi_char_mask.extract_multi_char_prompt_payload(result)
    assert payload["mask_render_version"] == (version or 1)
    assert payload["char_inform"] == ["girl, silver hair"]


@pytest.mark.parametrize("names", [("Shoko", "Hibiki"), ("Mira", "Ren")])
@pytest.mark.parametrize("mirror", [False, True])
def test_dynamic_default_starts_from_bodies_and_archives_keep_upper(names, mirror):
    regions = [dict(name=names[0], x=.05, y=.3, width=.4, height=.65),
               dict(name=names[1], x=.6, y=.2, width=.3, height=.7)]
    if mirror:
        for region in regions:
            region["x"] = 1 - region["x"] - region["width"]
    layout = multi_char_mask.validate_multi_char_layout({"regions": regions}, names)
    assert layout["mask_render_version"] == 5
    body = np.asarray(multi_char_mask.render_region_mask(layout, width=200, height=100))
    assert not body[0].any()
    for version in (3, 4):
        old = {**layout, "mask_render_version": version}
        upper = np.asarray(multi_char_mask.render_region_mask(old, width=200, height=100))
        assert upper[0].any()
        np.testing.assert_array_equal(upper[body.any(-1)], body[body.any(-1)])
        snapshot = multi_char_mask.normalize_multi_char_snapshot(
            {"enable": True, "character_order": old["character_order"], "layout": old})
        assert snapshot["layout"]["mask_render_version"] == version
        assert multi_char_mask.layout_fingerprint(old) != multi_char_mask.layout_fingerprint(layout)


@pytest.mark.parametrize("mirror", [False, True])
def test_detected_faces_attach_by_geometry_and_survive_reordering_and_occlusion(mirror):
    regions = [(.05, .3, .45, .7), (.6, .15, .3, .75)]
    boxes = [[8, 12, 30, 36], [65, 5, 82, 23]]
    if mirror:
        regions = [(1-x-w, y, w, h) for x, y, w, h in regions]
        boxes = [[100-c, b, 100-a, d] for a, b, c, d in boxes]
    assigned = tracking.associate(boxes[::-1], regions, {}, 100, 100)
    assert assigned == {0: boxes[0], 1: boxes[1]}
    assert tracking.associate(boxes, regions, assigned, 100, 100) == assigned
    assert tracking.associate([boxes[1]], regions, assigned, 100, 100) == {1: boxes[1]}
    assert tracking.associate([], regions, assigned, 100, 100) == {}


@pytest.mark.parametrize("shape", [(1, 10, 14), (1, 11, 15)])
def test_guards_fill_only_original_background_and_clear_device_references(shape):
    masks = [torch.zeros(1, 80, 112), torch.zeros(1, 80, 112)]
    masks[0][:, 32:, :48] = 1
    masks[1][:, 24:, 64:] = 1
    tracker = tracking.FaceTracker(masks, FORMAT, lambda image: [[4, 8, 35, 35], [70, 4, 98, 28]])
    tracker.observe(11, torch.zeros(1, 16, 1, 10, 14))
    assert tracker.epoch == 1
    h, w = (shape[-2]+1)//2, (shape[-1]+1)//2
    original = torch.zeros(3, h*w)
    original[0] = 1
    original[0, -2:] = 0
    original[1, -2] = 1
    original[2, -1] = 1
    build = lambda weights, *args: weights.clone()
    result, bias = tracker.layout(original, [2, 3, 4], shape, SimpleNamespace(patch_spatial=2),
                                  torch.device("cpu"), torch.float32, build)
    torch.testing.assert_close(result[:, -2:], original[:, -2:])
    torch.testing.assert_close(result.sum(0), torch.ones(h*w))
    assert result[1:, :-2].any()
    assert torch.equal(bias, result > 1e-6)
    previous = dict(tracker.faces)
    tracker.detector = lambda image: []
    tracker.observe(16, torch.zeros(1, 16, 1, 10, 14))
    assert tracker.faces == previous
    tracker.clear_device_cache()
    assert tracker.cached is None and tracker.cache_key is None
    assert isinstance(tracker.last_weights, np.ndarray)


@pytest.mark.parametrize("source_size", [(128, 128), (256, 192)])
@pytest.mark.parametrize("face, bounds", [
    pytest.param((56, 8, 72, 24), (3, 0, 5, 5), id="above"),
    pytest.param((56, 104, 72, 120), (3, 3, 5, 8), id="below"),
    pytest.param((8, 56, 24, 72), (0, 3, 5, 5), id="left"),
    pytest.param((104, 56, 120, 72), (3, 3, 8, 5), id="right"),
    pytest.param((8, 8, 24, 24), (0, 0, 5, 5), id="above-left"),
    pytest.param((104, 8, 120, 24), (3, 0, 8, 5), id="above-right"),
    pytest.param((8, 104, 24, 120), (0, 3, 5, 8), id="below-left"),
    pytest.param((104, 104, 120, 120), (3, 3, 8, 8), id="below-right"),
    pytest.param((56, 56, 72, 72), (3, 3, 5, 5), id="inside-keeps-source"),
    pytest.param((0, 0, 16, 16), (0, 0, 5, 5), id="clipped-padding"),
    pytest.param((16, 16, 112, 112), (0, 0, 8, 8), id="face-encloses-source"),
])
def test_face_and_source_form_filled_outer_box_at_any_source_resolution(source_size, face, bounds):
    source_w, source_h = source_size
    mask = torch.zeros(1, source_h, source_w)
    mask[:, source_h * 3 // 8:source_h * 5 // 8,
         source_w * 3 // 8:source_w * 5 // 8] = 1
    tracker = tracking.FaceTracker([mask], FORMAT, lambda image: [face])
    tracker.observe(4, torch.zeros(1, 16, 1, 16, 16))
    body = torch.zeros(8, 8)
    body[3:5, 3:5] = 1
    original = torch.stack((1 - body, body)).flatten(1)
    saved = original.clone()
    result, bias = tracker.layout(
        original, [2, 3], (1, 16, 16), SimpleNamespace(patch_spatial=2),
        torch.device("cpu"), torch.float32, lambda weights, *args: weights.clone())

    expected = torch.zeros(8, 8)
    left, top, right, bottom = bounds
    expected[top:bottom, left:right] = 1
    torch.testing.assert_close(result, torch.stack((1 - expected, expected)).flatten(1))
    torch.testing.assert_close(original, saved)
    assert torch.equal(bias, result > 1e-6)
    np.testing.assert_array_equal(tracker.last_weights[1], expected.numpy())


@pytest.mark.parametrize("subject", [0, 1])
@pytest.mark.parametrize("other_weight", [0.5, 1.0])
def test_outer_box_preserves_other_character_inside_the_bridge(subject, other_weight):
    body = torch.zeros(8, 8)
    body[3:5, 3:5] = 1
    other = torch.zeros(8, 8)
    other[2, 3:5] = other_weight
    regions = [body, other] if subject == 0 else [other, body]
    masks = [region.repeat_interleave(16, 0).repeat_interleave(16, 1).unsqueeze(0)
             for region in regions]
    tracker = tracking.FaceTracker(masks, FORMAT, lambda image: [])
    tracker.width = tracker.height = 128
    tracker.faces = {subject: (56, 8, 72, 24)}
    original = torch.stack((1 - body - other, *regions)).flatten(1)
    result, _ = tracker.layout(
        original, [2, 3, 4], (1, 16, 16), SimpleNamespace(patch_spatial=2),
        torch.device("cpu"), torch.float32, lambda weights, *args: weights.clone())

    expected = torch.zeros(8, 8)
    expected[:5, 3:5] = 1
    expected[2, 3:5] = 0
    torch.testing.assert_close(result[subject + 1].reshape(8, 8), expected)
    torch.testing.assert_close(result[:, other.flatten() > 0], original[:, other.flatten() > 0])
    torch.testing.assert_close(result.sum(0), torch.ones(64))


@pytest.mark.parametrize("edge_weight", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("mirror", [False, True])
def test_outer_box_completes_own_partial_edges_without_a_seam(edge_weight, mirror):
    body = torch.zeros(8, 8)
    body[3:6, 2:5] = edge_weight
    body[4, 3] = 1
    if mirror:
        body = body.flip(1)
    mask = body.repeat_interleave(16, 0).repeat_interleave(16, 1).unsqueeze(0)
    face = (72, 8, 88, 24) if mirror else (40, 8, 56, 24)
    tracker = tracking.FaceTracker([mask], FORMAT, lambda image: [face])
    tracker.observe(4, torch.zeros(1, 16, 1, 16, 16))
    original = torch.stack((1 - body, body)).flatten(1)
    result, _ = tracker.layout(
        original, [2, 3], (1, 16, 16), SimpleNamespace(patch_spatial=2),
        torch.device("cpu"), torch.float32, lambda weights, *args: weights.clone())
    expected = torch.zeros(8, 8)
    expected[:6, 2:5] = 1
    if mirror:
        expected = expected.flip(1)
    torch.testing.assert_close(result, torch.stack((1 - expected, expected)).flatten(1))
    torch.testing.assert_close(original, torch.stack((1 - body, body)).flatten(1))


def test_outer_box_recomputes_from_source_and_exports_applied_mask(tmp_path, monkeypatch):
    mask = torch.zeros(1, 128, 128)
    mask[:, 48:80, 48:80] = 1
    detections = iter([[], [(56, 8, 72, 24)], [(104, 56, 120, 72)], []])
    tracker = tracking.FaceTracker([mask], FORMAT, lambda image: next(detections))
    body = torch.zeros(8, 8)
    body[3:5, 3:5] = 1
    original = torch.stack((1 - body, body)).flatten(1)
    args = (original, [2, 3], (1, 16, 16), SimpleNamespace(patch_spatial=2),
            torch.device("cpu"), torch.float32, lambda weights, *args: weights.clone())
    x0 = torch.zeros(1, 16, 1, 16, 16)
    tracker.observe(4, x0)
    assert tracker.layout(*args) is None
    tracker.observe(11, x0)
    first, _ = tracker.layout(*args)
    assert first[1].reshape(8, 8)[2, 3] == 1
    tracker.observe(16, x0)
    second, _ = tracker.layout(*args)
    expected = torch.zeros(8, 8)
    expected[3:5, 3:] = 1
    torch.testing.assert_close(second, torch.stack((1 - expected, expected)).flatten(1))
    tracker.observe(27, x0)
    retained, _ = tracker.layout(*args)
    torch.testing.assert_close(retained, second)

    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_temp_directory=lambda: str(tmp_path)))
    exported = tracker.export((128, 128))
    from PIL import Image
    with Image.open(tmp_path / exported["subfolder"] / exported["filename"]) as image:
        pixels = np.asarray(image)
        np.testing.assert_array_equal(pixels[:, :, 0],
                                      expected.numpy().repeat(16, 0).repeat(16, 1) * 255)
        assert not pixels[:, :, 1:].any()


def test_observation_is_cpu_only_does_not_mutate_latents_and_disables_failed_detector():
    x = torch.randn(1, 16, 1, 10, 14)
    saved = x.clone()
    mask = torch.ones(1, 80, 112)
    seen = []
    def detect(image):
        seen.append(image.size)
        raise RuntimeError("detector unavailable")
    tracker = tracking.FaceTracker([mask], FORMAT, detect)
    tracker.observe(4, x)
    tracker.observe(11, x)
    assert seen == [(112, 80)]
    assert tracker.disabled and not tracker.faces
    torch.testing.assert_close(x, saved)
    batch_tracker = tracking.FaceTracker([mask], FORMAT, detect)
    batch_tracker.observe(4, x.repeat(2, 1, 1, 1, 1))
    assert batch_tracker.disabled and len(seen) == 1


def test_checkpoints_keep_schedule_and_allow_subsequent_application():
    assert tracking.observation_steps(30) == {4, 11, 16, 27}
    assert tracking.observation_steps(1) == set()
    for total in (2, 3, 8, 20, 50):
        assert len(tracking.observation_steps(total)) <= 4
        assert all(0 <= step < total-1 for step in tracking.observation_steps(total))


def test_final_mask_sidecar_is_cropped_and_never_overwrites_source(tmp_path, monkeypatch):
    tracker = tracking.FaceTracker([torch.ones(1, 80, 112)], FORMAT, lambda image: [])
    tracker.width, tracker.height = 1256, 864
    tracker.last_weights = np.zeros((2, 54, 79), dtype=np.float32)
    tracker.last_weights[1, :, 0] = 1
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_temp_directory=lambda: str(tmp_path)))
    first = tracker.export((1256, 864))
    second = tracker.export((1256, 864))
    assert first["filename"] != second["filename"]
    from PIL import Image
    with Image.open(tmp_path / first["subfolder"] / first["filename"]) as image:
        assert image.size == (1256, 864)
        assert image.getpixel((15, 0)) == (255, 0, 0)
        assert image.getpixel((16, 0)) == (0, 0, 0)


def test_step6_reads_dynamic_metadata_without_confusing_it_with_generated_image(monkeypatch):
    recorded, fetched = [], []
    async def fetch(filename, subfolder, kind, *, port):
        fetched.append((filename, subfolder, kind, port))
        return b"actual-final-mask"
    monkeypatch.setattr(illustration_flow, "record_image_mask", recorded.append)
    outputs = {"sampler": {"soya_dynamic_mask": [dict(filename="mask.png", subfolder="m", type="temp")]},
               "save": {"images": [dict(filename="image.png")]}}
    asyncio.run(illustration_flow.record_comfy_dynamic_mask(outputs, fetch, port=8188))
    assert fetched == [("mask.png", "m", "temp", 8188)]
    assert recorded == [b"actual-final-mask"]
    assert outputs["save"]["images"][0]["filename"] == "image.png"
    asyncio.run(illustration_flow.record_comfy_dynamic_mask({"save": outputs["save"]}, fetch, port=8188))
    assert len(recorded) == 1


def test_step6_missing_sidecar_preserves_finished_image(monkeypatch):
    recorded = []
    async def fetch(*args, **kwargs):
        raise OSError("missing temporary preview")
    monkeypatch.setattr(illustration_flow, "record_image_mask", recorded.append)
    asyncio.run(illustration_flow.record_comfy_dynamic_mask(
        {"1": {"soya_dynamic_mask": [{"filename": "gone.png"}]}}, fetch, port=8188))
    assert recorded == []
